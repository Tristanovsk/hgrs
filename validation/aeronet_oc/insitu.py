'''
AERONET-OC data of the matchups (``matchups.csv``): median of the records within ``LAG`` of the overpass
(nearest record within 3 h if there is none). Writes ``OUT_DIR/aeronet_at_overpass.nc`` (dimensions
``matchup``, ``wl_aeronet``):

* ``Rrs = Lwn_f/Q / F0`` (sr-1), F0 of Thuillier et al. (2003) averaged over 10 nm (Gaussian),
* ``aot`` spectrum, ``aot550``, ``aot865``, ``ang_exp`` (log-log interpolation),
* ``tcwv``, ``tco3``, ``tcno2`` (kg m-2), ``pressure`` (hPa), ``wind`` (m s-1), ``chl`` (mg m-3), ``sza_aeronet``,
  ``n_records`` and ``lag_min`` (time difference of the records to the overpass).

Usage: python insitu.py [lag_minutes]
'''
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from common import OUT_DIR, read_aeronet_oc

warnings.filterwarnings('ignore')
LAG = pd.Timedelta(minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 60)
F0_FILE = '/home/harmel/Dropbox/Dropbox/work/git/satellite_app/hgrs/hgrs/data/aux/Thuillier_2003_0.3nm.dat'
DU_O3, DU_NO2 = 2.14485e-5, 2.055482e-5  # DU -> kg m-2


def f0(wl):
    '''Thuillier (2003) solar irradiance (mW m-2 nm-1) averaged with a 10 nm Gaussian'''
    s = pd.read_csv(F0_FILE, skiprows=15, sep='\t', header=None, names=('wl', 'F0'))
    w = np.exp(-0.5 * ((s.wl.values[None, :] - np.asarray(wl)[:, None]) / (10 / 2.3548)) ** 2)
    return (w * s.F0.values).sum(1) / w.sum(1)


def spectral(df, field):
    '''Series (wavelength -> value) of a spectral field of a record'''
    sub = df.xs(field, level=1, axis=1)
    sub.columns = sub.columns.get_level_values(-1).astype(float)
    return sub


def loglog(wl, values, target):
    ok = np.isfinite(values) & (values > 0)
    if ok.sum() < 2:
        return np.nan
    return float(np.exp(np.interp(np.log(target), np.log(wl[ok]), np.log(values[ok]))))


def main():
    m = pd.read_csv(OUT_DIR / 'matchups.csv', parse_dates=['time', 'aeronet_time'])
    m = m.dropna(subset=['level'])
    sites = pd.read_csv(OUT_DIR / 'scenes_over_stations.csv').drop_duplicates('site')
    files = {}
    rows, spectra = [], []
    for r in m.itertuples():
        key = (r.site, r.level)
        if key not in files:
            import glob
            from common import AERONET_DIR
            files[key] = read_aeronet_oc(AERONET_DIR / f'{r.site}_OCv3.{r.level}')
        df = files[key]
        lag = np.abs(df.index - r.time)
        sel = df[lag <= LAG]
        if len(sel) == 0:
            sel = df.iloc[[int(np.argmin(lag))]]
        lags = np.abs(sel.index - r.time).total_seconds() / 60
        lwn = spectral(sel, 'Lwn_f/Q').median()
        aot = spectral(sel, 'Aerosol_Optical_Depth').median()
        wl = lwn.index.values
        scalar = lambda name: float(pd.to_numeric(sel.xs(name, level=1, axis=1).iloc[:, 0], errors='coerce').median())
        aot_wl, aot_v = aot.index.values, aot.values
        rows.append(dict(matchup=f'{r.product}__{r.site}', site=r.site, level=r.level, time=r.time,
                         n_records=len(sel), lag_min=float(np.median(lags)),
                         tcwv=10 * scalar('Total_Precipitable_Water(cm)'),
                         tco3=DU_O3 * scalar('Total_Ozone(Du)'), tcno2=DU_NO2 * scalar('Total_NO2(DU)'),
                         pressure=scalar('Pressure(hPa)'), wind=scalar('Wind_Speed(m/s)'),
                         chl=scalar('Chlorophyll-a'), sza_aeronet=scalar('Solar_Zenith_Angle'),
                         aot550=loglog(aot_wl, aot_v, 550.), aot865=loglog(aot_wl, aot_v, 865.)))
        rows[-1]['ang_exp'] = (-np.log(rows[-1]['aot550'] / rows[-1]['aot865']) / np.log(550 / 865.)
                               if rows[-1]['aot865'] > 0 else np.nan)
        spectra.append((pd.Series(lwn.values / (0.1 * f0(wl)), index=wl), aot))
    tab = pd.DataFrame(rows).set_index('matchup')
    ds = tab.to_xarray()
    wl_all = np.unique(np.concatenate([s[0].index.values for s in spectra] + [s[1].index.values for s in spectra]))
    rrs = np.full((len(rows), wl_all.size), np.nan)
    aot = np.full((len(rows), wl_all.size), np.nan)
    for i, (r_, a_) in enumerate(spectra):
        rrs[i, np.searchsorted(wl_all, r_.index.values)] = r_.values
        aot[i, np.searchsorted(wl_all, a_.index.values)] = a_.values
    ds['Rrs'] = (('matchup', 'wl_aeronet'), rrs)
    ds['aot'] = (('matchup', 'wl_aeronet'), aot)
    ds = ds.assign_coords(wl_aeronet=wl_all)
    ds['Rrs'].attrs.update(units='sr-1', description='Lwn_f/Q / F0 (Thuillier 2003, 10 nm)')
    for v, u in [('tcwv', 'kg m-2'), ('tco3', 'kg m-2'), ('tcno2', 'kg m-2'), ('pressure', 'hPa')]:
        ds[v].attrs['units'] = u
    ds.attrs['lag_max_min'] = LAG.total_seconds() / 60
    ds.to_netcdf(OUT_DIR / 'aeronet_at_overpass.nc')
    print(tab[['site', 'level', 'n_records', 'lag_min', 'tcwv', 'aot550', 'ang_exp', 'chl']].round(3).to_string())


if __name__ == '__main__':
    main()
