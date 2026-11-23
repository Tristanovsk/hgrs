'''
Validation of hGRS with the HYPERNETS hyperspectral radiometers (WATERHYPERNET network), EnMAP and PRISMA.

In situ data: the harmonised HYPERNETS file of the validation database (``HGRS_HYPERNETS``, default
``/data/insitu/validation_db/harmonised/hypernets.nc``: ``Rrs`` on 751 wavelengths, ``qc_flag``, geometry, per
sequence ``obs``), quality-approved sequences only (``qc_flag`` 0).

The processing reuses the AERONET-OC chain (``validation/aeronet_oc``): the matchups are written in the same
``matchups.csv`` format in their own output folder (``HGRS_VAL_OUT``), so that ``run_versions.py`` processes them
with each version of hGRS (crop around the station, ``runs/<version>/<product>__<site>.nc``).

Matchup protocol:

* scene covering the station (footprint of the L1 product), at least one QC-approved sequence within ``MAX_LAG``
  (60 min) of the overpass;
* in situ Rrs: mean and standard deviation of the QC-approved sequences within ``WINDOW`` (30 min; the nearest
  one within ``MAX_LAG`` if none), convolved with the Gaussian band responses of the sensor (centre and FWHM of
  the L1 metadata, ``fwhm_table``);
* satellite Rrs: median of the water pixels of the 5 x 5 pixel box at the station (``Rrs`` of the run output, see
  ``process_one.py``), with the quality checks of ``aeronet_oc/compare.py`` (number of water pixels, coefficient
  of variation at 560 nm).

Usage (``HGRS_VAL_SENSOR=enmap|prisma``, ``HGRS_VAL_OUT=<folder of the sensor>``)::

    python hypernets.py inventory           # matchups.csv
    python ../aeronet_oc/run_versions.py 1 <version>=<code>:<config> ...
    python hypernets.py compare <version> ...  # compare/hypernets_<version>.nc, compare/stats_hypernets.csv
'''
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'aeronet_oc'))
from common import OUT_DIR, SENSOR, scenes, inside, cams_file, image_path  # noqa: E402

warnings.filterwarnings('ignore')
HYPERNETS = Path(os.environ.get('HGRS_HYPERNETS', '/data/insitu/validation_db/harmonised/hypernets.nc'))
MAX_LAG = pd.Timedelta(minutes=60)
WINDOW = pd.Timedelta(minutes=30)
# sites without reliable coordinates in the harmonised file
SKIP_SITES = ('GAIT',)
QC = dict(n_pix_min=13, cv_max=0.20)


def load_insitu():
    '''QC-approved HYPERNETS sequences: Dataset (obs, wl) with site, time, Rrs, sza, vza, raa'''
    d = xr.open_dataset(HYPERNETS)
    site = np.array([str(s).split('_')[0] for s in d.subsource.values])
    keep = (d.qc_flag.values == 0) & ~np.isin(site, SKIP_SITES)
    d = d[['time', 'lat', 'lon', 'Rrs', 'sza', 'vza', 'raa']].isel(obs=np.flatnonzero(keep)).load()
    return d.assign(site=('obs', site[keep]))


def stations(insitu):
    df = pd.DataFrame(dict(site=insitu.site.values, lat=insitu.lat.values, lon=insitu.lon.values))
    return df.groupby('site').median()


def inventory():
    insitu = load_insitu()
    t = pd.to_datetime(insitu.time.values)
    rows = []
    for s in scenes().itertuples():
        for site, c in stations(insitu).iterrows():
            if not inside(c.lon, c.lat, s.footprint):
                continue
            lag = np.abs(t[insitu.site.values == site] - pd.Timestamp(s.time))
            if not len(lag) or lag.min() > MAX_LAG:
                continue
            rows.append(dict(product=s.product, path=s.path, version=s.version, time=s.time, site=site,
                             lon=c.lon, lat=c.lat, cams=cams_file(s.time), level='qc0',
                             lag_min=lag.min().total_seconds() / 60, n_records=int((lag <= MAX_LAG).sum())))
    m = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_DIR / 'matchups.csv', index=False)
    print(f'{len(m)} {SENSOR} matchups within {MAX_LAG}:', m.groupby('site').size().to_dict() if len(m) else {})


def fwhm_table(path):
    '''band centres and FWHM (nm) of the L1 product, sorted by wavelength'''
    p = image_path(path)
    if isinstance(p, list):
        import h5py
        with h5py.File(p[0], 'r') as f:
            cw = np.r_[f.attrs['List_Cw_Vnir'], f.attrs['List_Cw_Swir']]
            fw = np.r_[f.attrs['List_Fwhm_Vnir'], f.attrs['List_Fwhm_Swir']]
    else:
        import glob
        import xml.etree.ElementTree as ET
        root = ET.parse(glob.glob(os.path.join(p, '*METADATA.XML'))[0]).getroot()
        bands = root.find('specific').find('bandCharacterisation')
        cw = np.array([float(b.findtext('wavelengthCenterOfBand')) for b in bands])
        fw = np.array([float(b.findtext('FWHMOfBand')) for b in bands])
    ok = (cw > 0) & (fw > 0)
    i = np.argsort(cw[ok])
    return cw[ok][i], fw[ok][i]


def convolve(wl_in, rrs, centres, fwhm):
    '''hyperspectral Rrs (..., wl_in) -> Gaussian bands (..., band); NaN where the band is not covered'''
    out = np.full(rrs.shape[:-1] + (len(centres),), np.nan)
    for k, (c, f) in enumerate(zip(centres, fwhm)):
        s = f / 2.3548
        w = np.exp(-0.5 * ((wl_in - c) / s) ** 2)
        w[np.abs(wl_in - c) > 3 * f] = 0
        if c - f < wl_in.min() or c + f > wl_in.max() or w.sum() == 0:
            continue
        v = np.nansum(rrs * w, -1) / np.sum(np.where(np.isfinite(rrs), w, 0), -1)
        out[..., k] = v
    return out


def insitu_at_overpass(insitu, site, time):
    '''mean, std and number of the QC-approved sequences within WINDOW (nearest within MAX_LAG if none)'''
    sel = insitu.isel(obs=np.flatnonzero(insitu.site.values == site))
    lag = np.abs(pd.to_datetime(sel.time.values) - pd.Timestamp(time))
    idx = np.flatnonzero(lag <= WINDOW)
    if not len(idx):
        idx = [int(np.argmin(lag))] if lag.min() <= MAX_LAG else []
    if not len(idx):
        return None
    s = sel.isel(obs=idx)
    return dict(Rrs=s.Rrs.mean('obs').values, Rrs_std=s.Rrs.std('obs').values, n=len(idx),
                lag_min=float(lag[idx].min().total_seconds() / 60), sza=float(s.sza.mean()),
                vza=float(s.vza.mean()), raa=float(s.raa.mean()), wl=s.wl.values)


def compare(versions):
    insitu = load_insitu()
    m = pd.read_csv(OUT_DIR / 'matchups.csv', parse_dates=['time'])
    (OUT_DIR / 'compare').mkdir(exist_ok=True)
    rows = []
    for version in versions:
        data = []
        for r in m.itertuples():
            f = OUT_DIR / 'runs' / version / f'{r.product}__{r.site}.nc'
            if not f.exists():
                continue
            sat = xr.open_dataset(f).load()
            ins = insitu_at_overpass(insitu, r.site, r.time)
            if ins is None:
                continue
            cw, fw = fwhm_table(r.path)
            wl = sat.wl.values
            fwhm = np.interp(wl, cw, fw)
            rrs_ins = convolve(ins['wl'], ins['Rrs'], wl, fwhm)
            std_ins = convolve(ins['wl'], ins['Rrs_std'], wl, fwhm)
            i560 = int(np.argmin(np.abs(wl - 560)))
            n_pix = float(sat.n_pix) if 'n_pix' in sat else np.nan
            cv = float(sat.Rrs_std[i560] / sat.Rrs[i560]) if 'Rrs_std' in sat else np.nan
            qc = (n_pix >= QC['n_pix_min']) and (cv <= QC['cv_max'])
            data.append(xr.Dataset(dict(Rrs_sat=('wl', sat.Rrs.values), Rrs_insitu=('wl', rrs_ins),
                                        Rrs_insitu_std=('wl', std_ins), n_insitu=ins['n'],
                                        lag_min=ins['lag_min'], qc=bool(qc), n_pix=n_pix, cv560=cv,
                                        site=r.site, time=np.datetime64(r.time)),
                                   coords=dict(wl=wl)).expand_dims(matchup=[f'{r.product}__{r.site}']))
        if not data:
            print(version, 'no matchup processed')
            continue
        ds = xr.concat(data, 'matchup', join='outer')
        ds.to_netcdf(OUT_DIR / 'compare' / f'hypernets_{version}.nc')
        ok = ds.isel(matchup=np.flatnonzero(ds.qc.values))
        for w in (443, 490, 560, 620, 665, 709, 779, 865):
            k = int(np.argmin(np.abs(ok.wl.values - w)))
            x, y = ok.Rrs_insitu.values[:, k], ok.Rrs_sat.values[:, k]
            g = np.isfinite(x) & np.isfinite(y)
            if g.sum() < 2:
                continue
            d = y[g] - x[g]
            rows.append(dict(version=version, wl=float(ok.wl[k]), N=int(g.sum()), bias=np.median(d),
                             mad=np.median(np.abs(d)), mapd=100 * np.median(np.abs(d) / np.abs(x[g])),
                             r=np.corrcoef(x[g], y[g])[0, 1]))
        print(version, f'{ds.sizes["matchup"]} matchups, {int(ds.qc.sum())} pass the QC')
    if rows:
        st = pd.DataFrame(rows)
        st.to_csv(OUT_DIR / 'compare' / 'stats_hypernets.csv', index=False, float_format='%.5g')
        print(st.round(5).to_string(index=False))


if __name__ == '__main__':
    if sys.argv[1] == 'inventory':
        inventory()
    else:
        compare(sys.argv[2:])
