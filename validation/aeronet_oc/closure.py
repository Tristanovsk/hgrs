'''
Radiometric closure at the AERONET-OC stations: TOA reflectance simulated by radcalnet_oc from the AERONET-OC
measurements at the overpass (``aeronet_at_overpass.nc``) in the geometry of the EnMAP image, against the EnMAP
L1C TOA reflectance of the station (5 x 5 pixel median).

The simulation is that of radcalnet_oc (OPAC aerosol mixture retrieved from the spectral AOT, rough sea surface,
gaseous absorption at the resolution of the REPTRAN table), run with each gaseous transmittance model of
``GAS_COUPLINGS``, and convolved with Gaussian bands of the FWHM of each EnMAP band. The remote-sensing
reflectance of the water is that of the AERONET-OC bands (``Lwn_f/Q / F0``), interpolated linearly in wavelength
(constant below the first band, linearly decreasing to 0 from the last band to 1100 nm, 0 beyond): smooth
across the gaseous absorption bands, it is a source of error only in the visible and for turbid waters.

In the gaseous absorption bands, the simulation tests the gaseous transmittance (direct beam or coupled with the
scattering) for the water vapour column, the pressure and the ozone of AERONET; elsewhere, the radiometry of
EnMAP and the aerosol model.

Writes ``OUT_DIR/closure/<matchup>.nc`` (``Rtoa_enmap``, ``Rtoa_enmap_std``, ``Rtoa_<coupling>``,
``Ratm_<coupling>``, on the EnMAP bands) and ``OUT_DIR/closure/closure.nc`` (all the matchups).

Usage: python closure.py [n_workers] [matchup_filter]
'''
import logging
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from common import OUT_DIR

warnings.filterwarnings('ignore')
CLOSURE = OUT_DIR / 'closure'
# gaseous transmittance models: name -> (gas_coupling, o2_model) of radcalnet_oc, or ('hgrs', o2_model,
# coefficient table) for the transmittances of hgrs.gas_coupling.GasTransmittance (path-length model of the
# total optical thickness of each channel in the windows of the table)
HGRS_ROOT = Path(__file__).resolve().parents[2]
GAS_MODELS = {'decoupled': ('decoupled', 'lbl'), 'path_length': ('path_length', 'lbl'),
              'path_length_reptran': ('path_length', 'reptran'),
              'hgrs_t684': ('hgrs', 'lbl_o2b', HGRS_ROOT / 'hgrs' / 'data' / 'aux' / 'gas_coupling_ppld.csv'),
              'hgrs_full': ('hgrs', 'lbl_o2b', HGRS_ROOT / 'hgrs' / 'data' / 'aux' / 'gas_coupling_ppld_full.csv')}
WINDOW = 5


def enmap_station(path, lon, lat):
    '''EnMAP L1C TOA reflectance (median and std of a WINDOW x WINDOW box) and geometry at the station'''
    from pyproj import Transformer
    from hgrs.drivers.enmap import EnmapDriver
    ds = EnmapDriver().read(path)
    x, y = Transformer.from_crs(4326, ds.rio.crs, always_xy=True).transform(lon, lat)
    ix = int(np.argmin(np.abs(ds.x.values - x)))
    iy = int(np.argmin(np.abs(ds.y.values - y)))
    if np.hypot(ds.x.values[ix] - x, ds.y.values[iy] - y) > 60:
        raise RuntimeError('station outside the image')
    h = WINDOW // 2
    box = ds.isel(x=slice(ix - h, ix + h + 1), y=slice(iy - h, iy + h + 1))[['Rtoa', 'sza', 'vza', 'raa']].load()
    return xr.Dataset(dict(Rtoa=box.Rtoa.median(['x', 'y']), Rtoa_std=box.Rtoa.std(['x', 'y']), fwhm=ds.fwhm.load(),
                           sza=box.sza.median(), vza=box.vza.median(), raa=box.raa.median())).reset_coords(drop=True)


def hyperspectral_rrs(wl_aer, rrs_aer, wl):
    '''Rrs on ``wl`` from the AERONET-OC bands (see the module docstring)'''
    ok = np.isfinite(rrs_aer)
    w, r = np.asarray(wl_aer)[ok], np.clip(np.asarray(rrs_aer)[ok], 0, None)
    out = np.interp(wl, w, r)
    tail = wl > w[-1]
    out[tail] = r[-1] * np.clip((1100. - wl[tail]) / (1100. - w[-1]), 0, 1)
    return out


def gaussian_bands(values, wl_fine, centers, fwhms):
    '''values (..., wl_fine) -> Gaussian bands (..., band)'''
    out = []
    for c, f in zip(centers, fwhms):
        s = f / 2.3548
        w = np.exp(-0.5 * ((wl_fine - c) / s) ** 2)
        w[np.abs(wl_fine - c) > 3 * f] = 0
        out.append((values * w).sum(-1) / w.sum())
    return np.stack(out, -1)


def hgrs_gas_transmittance(o2_model, table):
    '''replacement of radcalnet_oc Process.get_gas_transmittance with the transmittances of hGRS (terms 'Ed',
    'atm' and 'w' of hgrs.gas_coupling.GasTransmittance, coefficient table ``table``)'''
    sys.path.insert(0, str(HGRS_ROOT))
    from hgrs.gas_coupling import GasTransmittance

    def get_gas_transmittance(self):
        db = self.input_db
        gt = GasTransmittance(self.gas_trans.gas_lut, 1., 'path_length', o2_model, ppld_file=str(table))
        wl = self.gas_trans.gas_lut.wl
        # one time (the input is duplicated, see simulate), viewing angles of muv
        i0 = dict(time=0)
        args = lambda m: (m, float(db.pressure.isel(**i0)), float(db.tco3.isel(**i0)), float(db.tcno2.isel(**i0)),
                          0.0105, float(db.tcwv.isel(**i0)), float(self.aot550.isel(**i0)))
        m0 = 1. / float(self.mu0.isel(**i0))
        muv = np.atleast_1d(self.muv.values)
        times = db.time
        def wrap(t, with_vza):
            if with_vza:
                da = xr.DataArray(t, coords={'vza': self.vza.values, 'wl': wl.values}, dims=('vza', 'wl'))
            else:
                da = xr.DataArray(t, coords={'wl': wl.values}, dims='wl')
            return da.expand_dims(time=times.values)
        self.Tg_Ed = wrap(gt.total('Ed', *args(m0)), False)
        self.Tg_atm = wrap(np.stack([gt.total('atm', *args(m0 + 1. / u)) for u in muv]), True)
        self.Tg_w = wrap(np.stack([gt.total('w', *args(m0 + 1. / u)) for u in muv]), True)
    return get_gas_transmittance


def simulate(aer, enmap, coupling, o2_model='lbl', table=None):
    '''radcalnet_oc TOA simulation (full resolution) for one matchup and gaseous transmittance model'''
    import radcalnet_oc as radoc
    hgrs_model = coupling == 'hgrs'
    if hgrs_model:
        coupling, o2_model_hgrs = 'decoupled', o2_model
        o2_model = 'lbl'
    logging.getLogger().setLevel(logging.WARNING)
    t = pd.Timestamp(aer.time.values)
    wl_rrs = np.arange(350., 2501., 1.)
    rrs = hyperspectral_rrs(aer.wl_aeronet.values, aer.Rrs.values, wl_rrs)
    aot = aer.aot.dropna('wl_aeronet').reset_coords(drop=True)
    # the measurement is given twice (1 s apart): radcalnet_oc squeezes its LUT, which fails with one time
    times = [t, t + pd.Timedelta(seconds=1)]
    naot = (aot / float(aer.aot550)).drop_vars('time', errors='ignore').expand_dims(time=times)
    two = lambda v: ('time', [v, v])
    inp = xr.Dataset(
        dict(tcwv=two(float(aer.tcwv)), tco3=two(float(aer.tco3)), tcno2=two(float(aer.tcno2)),
             pressure=two(float(aer.pressure) if np.isfinite(aer.pressure) else 1013.25),
             sza=two(float(enmap.sza)), day_of_year=two(t.dayofyear),
             aot550=two(float(aer.aot550)), aot865=two(float(aer.aot865)), ang_exp=two(float(aer.ang_exp)),
             Rrs=(('time', 'wl'), np.stack([rrs, rrs]))),
        coords=dict(time=times, wl=wl_rrs))
    inp['naot'] = naot
    # relative azimuth of radcalnet_oc: 0 deg towards the Sun (hGRS: raa_lut = (180 - raa) % 360)
    azi = (180. - float(enmap.raa)) % 360.
    azi = 360. - azi if azi > 180 else azi
    vza = xr.DataArray([float(enmap.vza)], coords={'vza': [float(enmap.vza)]}, dims='vza')
    proc = radoc.Process(inp, vza=vza, azi=xr.DataArray([azi], coords={'azi': [azi]}, dims='azi'),
                         gas_coupling=coupling, o2_model=o2_model)
    if hgrs_model:
        import types
        proc.get_gas_transmittance = types.MethodType(hgrs_gas_transmittance(o2_model_hgrs, table), proc)
    aero = radoc.Aerosol(naot, proc.lut.naot_lut)
    aero.process()
    proc.execute(aerosol_combination=aero.model_db.aerosol_combination)
    db = proc.radcalnet_db.isel(time=0).squeeze()
    return db, aero.model_db.aerosol_combination.isel(time=0).squeeze()


def available(name):
    '''False for an hGRS model whose coefficient table does not exist'''
    m = GAS_MODELS[name]
    return m[0] != 'hgrs' or Path(m[2]).exists()


def run(row):
    out_file = CLOSURE / f'{row.matchup}.nc'
    try:
        if out_file.exists():
            # add the gaseous transmittance models missing from an existing file
            out = xr.open_dataset(out_file).load()
            todo = [k for k in GAS_MODELS if f'Rtoa_{k}' not in out and available(k)]
            if not todo:
                return row.matchup, 'exists'
            enmap = out[['sza', 'vza', 'raa']].assign_coords(wl=out.wl)
            enmap['fwhm'] = out.fwhm
        else:
            aer = xr.open_dataset(OUT_DIR / 'aeronet_at_overpass.nc').sel(matchup=row.matchup).load()
            if not (np.isfinite(aer.aot550) and np.isfinite(aer.tcwv)):
                return row.matchup, 'no AOT or water vapour'
            enmap = enmap_station(row.path, row.lon, row.lat)
            out = xr.Dataset(dict(Rtoa_enmap=enmap.Rtoa, Rtoa_enmap_std=enmap.Rtoa_std, fwhm=enmap.fwhm))
            for k in ('sza', 'vza', 'raa'):
                out[k] = enmap[k]
            for k in ('tcwv', 'tco3', 'pressure', 'aot550', 'ang_exp', 'site'):
                out[k] = aer[k].reset_coords(drop=True)
            out.attrs.update(matchup=row.matchup, product=row.product)
            todo = [k for k in GAS_MODELS if available(k)]
        aer = xr.open_dataset(OUT_DIR / 'aeronet_at_overpass.nc').sel(matchup=row.matchup).load()
        for name in todo:
            db, comb = simulate(aer, enmap, *GAS_MODELS[name])
            wl_f = db.wl.values
            for v in ('Rtoa', 'Ratm'):
                vals = db[v].transpose(..., 'wl').values
                out[f'{v}_{name}'] = ('wl', gaussian_bands(vals, wl_f, enmap.wl.values, enmap.fwhm.values))
        out['aerosol_combination'] = comb.reset_coords(drop=True)
        out.to_netcdf(str(out_file) + '.tmp')
        Path(str(out_file) + '.tmp').replace(out_file)
        return row.matchup, 'ok ' + ' '.join(todo)
    except Exception as e:
        return row.matchup, repr(e)[:200]


def main(n_workers=4, pattern=''):
    CLOSURE.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(OUT_DIR / 'matchups.csv')
    m['matchup'] = m['product'] + '__' + m.site
    m = m[m.matchup.str.contains(pattern, regex=False)]
    rows = [SimpleNamespace(**r) for r in m.to_dict('records')]
    with Pool(n_workers) as pool:
        for mid, status in pool.imap_unordered(run, rows):
            print(status, mid, flush=True)
    files = sorted(CLOSURE.glob('ENMAP01*.nc'))
    data = []
    for f in files:
        ds = xr.open_dataset(f).load()
        ds = ds.assign_coords(band=('wl', np.arange(ds.sizes['wl']))).swap_dims(wl='band').rename(wl='wl_band')
        data.append(ds.expand_dims(matchup=[ds.attrs['matchup']]))
    if data:
        xr.concat(data, 'matchup', coords='different', compat='equals', join='outer',
                  combine_attrs='drop').to_netcdf(CLOSURE / 'closure.nc')
        print(f'{len(data)} matchups -> {CLOSURE / "closure.nc"}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4, sys.argv[2] if len(sys.argv) > 2 else '')
