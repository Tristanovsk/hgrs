'''
AERONET-OC data of the matchups (``matchups.csv``): median of the records within ``LAG`` of the overpass
(nearest record within 3 h if there is none), read from the AERONET-OC database (``aeronet_oc_db``,
netCDF products of ``HGRS_VAL_AERONET/db``). Writes ``OUT_DIR/aeronet_at_overpass.nc`` (dimensions
``matchup``, ``wl_aeronet``):

* ``Rrs = Lwn_f/Q / F0`` (sr-1), F0 of Thuillier et al. (2003) averaged over 10 nm (Gaussian) at the exact
  wavelength of the band (``Rrs_Lwn_fQ`` of the database),
* ``aot`` spectrum, ``aot550``, ``aot865``, ``ang_exp`` (log-log interpolation),
* ``tcwv``, ``tco3``, ``tcno2`` (kg m-2), ``pressure`` (hPa), ``wind`` (m s-1), ``chl`` (mg m-3), ``sza_aeronet``,
  ``n_records`` and ``lag_min`` (time difference of the records to the overpass).

Usage: python insitu.py [lag_minutes]
'''
import sys
import warnings

import numpy as np
import pandas as pd

from common import OUT_DIR, aeronet_db

warnings.filterwarnings('ignore')
LAG = pd.Timedelta(minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 60)
FALLBACK_LAG = pd.Timedelta(hours=3)
DU_O3, DU_NO2 = 2.14485e-5, 2.055482e-5  # DU -> kg m-2


def loglog(wl, values, target):
    ok = np.isfinite(values) & (values > 0)
    if ok.sum() < 2:
        return np.nan
    return float(np.exp(np.interp(np.log(target), np.log(wl[ok]), np.log(values[ok]))))


def at_overpass(db, site, level, times):
    '''database statistics at the overpass times of a site: dict time -> Dataset of one match'''
    out = {}
    res = db.match(site, times, level=level, max_lag=LAG, stat='median')
    for k in range(res.sizes.get('match', 0)):
        out[pd.Timestamp(res.time.values[k])] = res.isel(match=k)
    missing = [t for t in times if t not in out]
    if missing:
        res = db.match(site, missing, level=level, max_lag=FALLBACK_LAG, stat='nearest')
        for k in range(res.sizes.get('match', 0)):
            out[pd.Timestamp(res.time.values[k])] = res.isel(match=k)
    return out


def main():
    m = pd.read_csv(OUT_DIR / 'matchups.csv', parse_dates=['time', 'aeronet_time'])
    m = m.dropna(subset=['level'])
    db = aeronet_db()
    stats = {}
    for (site, level), g in m.groupby(['site', 'level']):
        stats[(site, level)] = at_overpass(db, site, level, pd.DatetimeIndex(g.time.unique()))

    rows, spectra = [], []
    for r in m.itertuples():
        s = stats[(r.site, r.level)].get(pd.Timestamp(r.time))
        if s is None:
            print(f'no AERONET-OC record within {FALLBACK_LAG} for {r.product} {r.site}')
            continue
        wl = s.wl.values
        rrs, aot = s.Rrs_Lwn_fQ.values, s.aot.values
        scalar = lambda name: float(s[name]) if name in s else np.nan
        rows.append(dict(matchup=f'{r.product}__{r.site}', site=r.site, level=r.level, time=r.time,
                         n_records=int(s.n_records), lag_min=float(s.lag_min),
                         tcwv=10 * scalar('tcwv'), tco3=DU_O3 * scalar('tco3'), tcno2=DU_NO2 * scalar('tno2'),
                         pressure=scalar('pressure'), wind=scalar('wind'), chl=scalar('chl'),
                         sza_aeronet=scalar('sza'),
                         aot550=loglog(wl, aot, 550.), aot865=loglog(wl, aot, 865.)))
        rows[-1]['ang_exp'] = (-np.log(rows[-1]['aot550'] / rows[-1]['aot865']) / np.log(550 / 865.)
                               if rows[-1]['aot865'] > 0 else np.nan)
        spectra.append((pd.Series(rrs, index=wl), pd.Series(aot, index=wl)))
    tab = pd.DataFrame(rows).set_index('matchup')
    ds = tab.to_xarray()
    wl_all = np.unique(np.concatenate([s[0].index.values for s in spectra]))
    rrs = np.full((len(rows), wl_all.size), np.nan)
    aot = np.full((len(rows), wl_all.size), np.nan)
    for i, (r_, a_) in enumerate(spectra):
        rrs[i, np.searchsorted(wl_all, r_.index.values)] = r_.values
        aot[i, np.searchsorted(wl_all, a_.index.values)] = a_.values
    ds['Rrs'] = (('matchup', 'wl_aeronet'), rrs)
    ds['aot'] = (('matchup', 'wl_aeronet'), aot)
    ds = ds.assign_coords(wl_aeronet=wl_all)
    ds['Rrs'].attrs.update(units='sr-1', description='Lwn_f/Q / F0 (Thuillier 2003, 10 nm, exact band wavelength)')
    for v, u in [('tcwv', 'kg m-2'), ('tco3', 'kg m-2'), ('tcno2', 'kg m-2'), ('pressure', 'hPa')]:
        ds[v].attrs['units'] = u
    ds.attrs['lag_max_min'] = LAG.total_seconds() / 60
    ds.attrs['aeronet_source'] = str(db.dir)
    ds.to_netcdf(OUT_DIR / 'aeronet_at_overpass.nc')
    print(tab[['site', 'level', 'n_records', 'lag_min', 'tcwv', 'aot550', 'ang_exp', 'chl']].round(3).to_string())


if __name__ == '__main__':
    main()
