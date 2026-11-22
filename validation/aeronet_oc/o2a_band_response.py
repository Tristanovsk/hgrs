'''
O2-A band of EnMAP: fit of the band responses on the closure at the AERONET-OC stations (see closure.py).

``simulate`` runs the radcalnet_oc simulations of closure.py again for the matchups of
``closure/band_depth_ratio.csv`` and keeps the full-resolution TOA and path reflectances over 730-800 nm (~0.06 nm)
for three gaseous transmittance models (``closure/fine/<matchup>.nc``). ``fit`` convolves them with Gaussian bands of
the EnMAP metadata shifted by ``dc`` and widened by ``df`` (756, 763.7 and 771 nm bands only), divides by the EnMAP
reflectance and by a linear continuum through the 741, 749, 778 and 786 nm bands, and searches the common
(``dc``, ``df``) that makes the median residuals of the three bands smallest, for each model, and the change of the
763.7 nm band alone (``closure/srf_fit.csv``).

Usage::

    python o2a_band_response.py simulate [matchup ...]    # ~25 s per matchup, < 1 GB
    python o2a_band_response.py fit
'''
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from common import OUT_DIR

warnings.filterwarnings('ignore')
CLOSURE = OUT_DIR / 'closure'
FINE = CLOSURE / 'fine'
MODELS = ['path_length', 'decoupled', 'path_length_reptran']
CONT = [741.5, 748.8, 778.3, 785.8]
ABS = [756.1, 763.4, 770.8]


def simulate(matchups):
    import closure
    FINE.mkdir(parents=True, exist_ok=True)
    if not matchups:
        matchups = pd.read_csv(CLOSURE / 'band_depth_ratio.csv').matchup.unique()
    for mid in matchups:
        out = xr.open_dataset(CLOSURE / f'{mid}.nc').load()
        enmap = out[['sza', 'vza', 'raa']].assign_coords(wl=out.wl)
        enmap['fwhm'] = out.fwhm
        aer = xr.open_dataset(OUT_DIR / 'aeronet_at_overpass.nc').sel(matchup=mid).load()
        res = {}
        for name in MODELS:
            db, _ = closure.simulate(aer, enmap, *closure.GAS_MODELS[name])
            sel = db.sel(wl=slice(730, 800))
            for v in ('Rtoa', 'Ratm'):
                res[f'{v}_{name}'] = ('wl', sel[v].transpose(..., 'wl').values.reshape(-1, sel.sizes['wl'])[0])
            wl = sel.wl.values
        xr.Dataset(res, coords=dict(wl=wl)).to_netcdf(FINE / f'{mid}.nc')
        print(mid, flush=True)


def gaussian(wl, v, c, f):
    s = f / 2.3548
    w = np.exp(-0.5 * ((wl - c) / s) ** 2)
    w[np.abs(wl - c) > 3 * f] = 0
    return (v * w).sum() / w.sum()


def load():
    ds = xr.open_dataset(CLOSURE / 'closure.nc').load()
    data = []
    for fn in sorted(FINE.glob('*.nc')):
        fine = xr.open_dataset(fn).load()
        x = ds.sel(matchup=fn.stem)
        w = x.wl_band.values
        idx = {b: int(np.argmin(abs(w - b))) for b in CONT + ABS}
        data.append((fine, {b: (w[i], float(x.fwhm[i]), float(x.Rtoa_enmap[i])) for b, i in idx.items()}))
    return data


def residuals(data, model, dc, df, only=None):
    '''sim / EnMAP / linear continuum - 1 at the ABS bands (matchup, band)'''
    out = []
    for fine, en in data:
        wl, v = fine.wl.values, fine[f'Rtoa_{model}'].values
        r = {}
        for b in CONT + ABS:
            c, f, e = en[b]
            change = b in ABS and (only is None or b == only)
            r[b] = gaussian(wl, v, c + dc * change, f + df * change) / e
        p = np.polyfit([en[b][0] for b in CONT], [r[b] for b in CONT], 1)
        out.append([r[b] / np.polyval(p, en[b][0]) - 1 for b in ABS])
    return np.array(out)


def fit():
    data = load()
    print(len(data), 'matchups')
    rows = []
    for m in MODELS:
        print(f'\n{m}: residual (%) 756/763/771 with the metadata bands:',
              np.round(100 * np.median(residuals(data, m, 0, 0), 0), 1))
        for dc in np.arange(-4, 4.01, 0.25):
            for df in np.arange(-4, 4.01, 0.5):
                med = np.median(residuals(data, m, dc, df), 0)
                rows.append(dict(model=m, bands='all', dc=dc, df=df, cost=np.sum(med ** 2),
                                 **{f'r{b}': 100 * med[i] for i, b in enumerate(ABS)}))
            for df in (-2, -1, 0, 1, 2):
                med = np.median(residuals(data, m, dc, df, only=763.4), 0)
                rows.append(dict(model=m, bands='763', dc=dc, df=df, cost=med[1] ** 2,
                                 **{f'r{b}': 100 * med[i] for i, b in enumerate(ABS)}))
    t = pd.DataFrame(rows)
    t.to_csv(CLOSURE / 'srf_fit.csv', index=False, float_format='%.4g')
    best = t.loc[t.groupby(['model', 'bands']).cost.idxmin()]
    print('\nbest band changes (nm) and residuals (%):')
    print(best.drop(columns='cost').round(2).to_string(index=False))


if __name__ == '__main__':
    if sys.argv[1] == 'simulate':
        simulate(sys.argv[2:])
    else:
        fit()
