r'''
Statistics of the radiometric closure (``closure/closure.nc``, see closure.py) in the gaseous absorption bands.

The ratio :math:`\rho(\lambda) = R_{sim}(\lambda) / R_{EnMAP}(\lambda)` of each matchup is divided by a smooth
function of the wavelength (polynomial of degree ``DEG`` in :math:`\ln\lambda`, fitted on :math:`\ln\rho` at the
bands of ``CONTINUUM``, where the gaseous absorption is weak): this removes the smooth differences (aerosol model,
water-leaving reflectance, calibration), and the residual :math:`\rho / \rho_{smooth} - 1` in the absorption bands
is the relative error of the simulated gaseous transmittance of the TOA signal (with the noise of the
measurement). The same is done for the band depth relative to the neighbouring bands (``DEPTHS``).

Writes ``closure/residual_by_band.csv`` (median and spread of the residual per EnMAP band and model) and
``closure/band_depth_ratio.csv``.

Usage: python closure_stats.py [model ...]
'''
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from common import OUT_DIR

warnings.filterwarnings('ignore')
CLOSURE = OUT_DIR / 'closure'
# bands with weak gaseous absorption (nm, nearest EnMAP band)
CONTINUUM = [443, 490, 560, 665, 709, 745, 778, 865, 1040, 1050, 1235, 1245, 1555, 1700, 1750, 2125, 2215]
DEG = 3
# absorption band, left and right neighbours (nm)
DEPTHS = {'O2-B 687': (687., 676., 704.), 'H2O 720': (720.5, 706.6, 741.8), 'O2-A 764': (763.7, 749.1, 778.6),
          'H2O 817': (816.6, 778.6, 863.7), 'H2O 900': (900., 863.7, 1003.9), 'H2O 941': (941.2, 863.7, 1003.9),
          'H2O 1130': (1128., 1040., 1235.), 'O2 1268': (1268., 1245., 1290.),
          'CO2 1575': (1575., 1555., 1625.), 'CO2 1605': (1605., 1555., 1625.), 'CH4 1665': (1665., 1625., 1700.),
          'CO2 2010': (2010., 2125., 2215.), 'CO2 2060': (2060., 2125., 2215.), 'CH4 2300': (2300., 2215., 2380.)}
# quality: AERONET record close to the overpass, uniform box at the station
QC = dict(cv_max=0.15, aot_max=1.0)


def nearest(wl, w):
    return int(np.argmin(np.abs(wl - w)))


def smooth_ratio(wl, rho):
    '''smooth part of the ratio (polynomial in ln wl fitted on ln rho at the CONTINUUM bands)'''
    idx = [nearest(wl, w) for w in CONTINUUM]
    x, y = np.log(wl[idx]), np.log(rho[idx])
    ok = np.isfinite(y)
    if ok.sum() < DEG + 2:
        return np.full_like(wl, np.nan)
    c = np.polyfit(x[ok], y[ok], DEG)
    return np.exp(np.polyval(c, np.log(wl)))


def depth_ratio(wl, r, band, left, right):
    '''R(band) / linear continuum of R(left), R(right)'''
    b, l, rr = (nearest(wl, w) for w in (band, left, right))
    cont = r[l] + (r[rr] - r[l]) * (wl[b] - wl[l]) / (wl[rr] - wl[l])
    return r[b] / cont


def load():
    '''all the matchups of closure/ (per-matchup files, not the aggregate written by closure.py)'''
    data = []
    for f in sorted(CLOSURE.glob('ENMAP01*.nc')):
        with xr.open_dataset(f) as ds:
            ds = ds.load()
        ds = ds.assign_coords(band=('wl', np.arange(ds.sizes['wl']))).swap_dims(wl='band').rename(wl='wl_band')
        data.append(ds.expand_dims(matchup=[ds.attrs['matchup']]))
    return xr.concat(data, 'matchup', coords='different', compat='equals', join='outer', combine_attrs='drop')


def main(models):
    ds = load()
    models = models or [v[5:] for v in ds.data_vars if v.startswith('Rtoa_') and v not in ('Rtoa_enmap', 'Rtoa_enmap_std')]
    wl_all = ds.wl_band.values           # (matchup, band)
    keep = []
    for i, m in enumerate(ds.matchup.values):
        d = ds.isel(matchup=i)
        cv = float((d.Rtoa_enmap_std / d.Rtoa_enmap).sel(band=nearest(d.wl_band.values, 865)))
        if cv <= QC['cv_max'] and float(d.aot550) <= QC['aot_max'] and np.isfinite(d.Rtoa_enmap).sum() > 100:
            keep.append(i)
    print(f'{ds.sizes["matchup"]} matchups, {len(keep)} pass the quality checks (cv(865) <= {QC["cv_max"]})')
    res_rows, dep_rows = [], []
    for i in keep:
        d = ds.isel(matchup=i)
        wl = d.wl_band.values
        enmap = d.Rtoa_enmap.values
        for mod in models:
            if f'Rtoa_{mod}' not in d or not np.isfinite(d[f'Rtoa_{mod}']).any():
                continue
            sim = d[f'Rtoa_{mod}'].values
            rho = sim / enmap
            res = rho / smooth_ratio(wl, rho) - 1
            for b in range(len(wl)):
                res_rows.append(dict(model=mod, band=b, wl=wl[b], matchup=str(d.matchup.values), residual=res[b]))
            for name, (b, l, r) in DEPTHS.items():
                if wl.max() < r:
                    continue
                dep_rows.append(dict(model=mod, band=name, matchup=str(d.matchup.values),
                                     enmap=depth_ratio(wl, enmap, b, l, r), sim=depth_ratio(wl, sim, b, l, r),
                                     tcwv=float(d.tcwv), aot550=float(d.aot550), site=str(d.site.values)))
    res = pd.DataFrame(res_rows)
    dep = pd.DataFrame(dep_rows)
    dep['err'] = dep.sim - dep.enmap
    g = res.groupby(['model', 'band'])
    summ = g.residual.agg(median='median', p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75),
                          N='count').join(g.wl.median())
    summ.reset_index().to_csv(CLOSURE / 'residual_by_band.csv', index=False, float_format='%.5g')
    dep.to_csv(CLOSURE / 'band_depth_ratio.csv', index=False, float_format='%.5g')
    pd.set_option('display.width', 220)
    print('\nband / continuum ratio: EnMAP and error of the simulation (sim - EnMAP), median [p25, p75]')
    t = dep.groupby(['band', 'model']).agg(N=('err', 'count'), enmap=('enmap', 'median'), sim=('sim', 'median'),
                                           err=('err', 'median'), mad=('err', lambda x: np.median(np.abs(x))))
    order = list(DEPTHS)
    t = t.reset_index()
    t['band'] = pd.Categorical(t.band, order, ordered=True)
    print(t.sort_values(['band', 'model']).round(3).to_string(index=False))
    print('\nrelative residual of the simulated TOA reflectance after removal of the smooth ratio (median, %)')
    piv = summ.reset_index().pivot_table(index='wl', columns='model', values='median')
    absorbing = piv[(piv.abs() > 0.01).any(axis=1)]
    print((100 * absorbing).round(1).to_string())
    print('\nmedian |residual| over all the bands (%):', (100 * res.groupby('model').residual
                                                       .apply(lambda x: np.nanmedian(np.abs(x)))).round(2).to_dict())


if __name__ == '__main__':
    main(sys.argv[1:])
