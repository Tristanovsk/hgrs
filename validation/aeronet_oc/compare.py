'''
Comparison of the hGRS versions with AERONET-OC (``aeronet_at_overpass.nc`` and ``runs/<version>/*.nc``).

* Rrs: EnMAP spectrum of the station (median of the water pixels of the 5 x 5 pixel core) weighted by a Gaussian
  of 10 nm FWHM centred on each AERONET-OC band (bands of EnMAP within 2 FWHM, NaN skipped);
* AOT at 550 nm and water vapour column of the mega-pixel of the station against AERONET-OC;
* band depth of Rrs in the absorption bands (Rrs of the band minus the linear continuum of its neighbours), at the
  station and for the darkest water pixels of the crop: a correct gaseous correction gives a smooth spectrum.

Only the matchups processed by all the versions and passing the quality checks (``QC``) are compared. AERONET-OC
Rrs below ``QC['rrs_min']`` (-0.005 sr-1) are not physical (a few values of -0.04 to -0.14 sr-1 at 779-1020 nm,
far from the noise of clear water, within 3e-4 sr-1 of 0) and are set to NaN (band left out of the matchup).
When DLR has processed a scene several times, only the latest processing is kept (``latest_processing``), so that
each overpass counts once.
Writes ``OUT_DIR/compare/`` : ``matchups_<version>.nc`` (matchup x AERONET band), ``stats_rrs.csv``,
``stats_params.csv``, ``band_depth.csv`` and figures.

Usage: python compare.py version [version ...]
'''
import re
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from common import OUT_DIR

warnings.filterwarnings('ignore')
FWHM = 10.
CMP = OUT_DIR / 'compare'
# band, left and right neighbours (nm): O2-B, H2O 720, O2-A, H2O 820, H2O 940, H2O 1130, O2 1270
DEPTHS = [(687.0, 679.7, 699.8), (720.5, 706.6, 741.8), (763.7, 749.1, 778.6), (816.6, 778.6, 863.7),
          (941.2, 887.7, 993.1), (1135., 1050., 1240.), (1268., 1240., 1290.)]
QC = dict(n_pix_min=13, cv_max=0.25, lag_max=60., rrs_min=-0.005)


def valid_aeronet(aer):
    '''AERONET-OC data with the non-physical Rrs (< QC['rrs_min']) set to NaN; number of values removed'''
    bad = aer.Rrs < QC['rrs_min']
    return aer.assign(Rrs=aer.Rrs.where(~bad)), int(bad.sum())


# EnMAP product: scene (datatake, start time, tile) and DLR processing time
ENMAP_PRODUCT = re.compile(r'^(ENMAP01-_+L\w+-DT\d+_\d{8}T\d{6}Z_\d{3})_V\d+_(\d{8}T\d{6}Z)')


def latest_processing(matchups):
    '''matchups (``<product>__<site>``) without the older DLR processings of the same scene and site; other
    products (e.g. PRISMA) are kept as they are'''
    latest = {}
    for m in matchups:
        product, _, site = m.rpartition('__')
        g = ENMAP_PRODUCT.match(product)
        key, proc = ((g[1], site), g[2]) if g else ((product, site), '')
        if key not in latest or proc > latest[key][0]:
            latest[key] = (proc, m)
    keep = {m for _, m in latest.values()}
    return [m for m in matchups if m in keep]


def to_aeronet_bands(rrs, wl_aer):
    '''Rrs (..., wl) -> AERONET bands (Gaussian of FWHM nm)'''
    wl = rrs.wl.values
    sig = FWHM / 2.3548
    w = np.exp(-0.5 * ((wl[None, :] - wl_aer[:, None]) / sig) ** 2)
    w[np.abs(wl[None, :] - wl_aer[:, None]) > FWHM] = 0
    v = rrs.values
    ok = np.isfinite(v)
    num = np.where(ok, v, 0) @ w.T
    den = ok.astype(float) @ w.T
    # require at least half of the weight (NaN where no band of the sensor is within FWHM, e.g. 400 nm for EnMAP)
    out = np.where((den > 0) & (den >= 0.5 * w.sum(1)), num / np.where(den > 0, den, 1), np.nan)
    return xr.DataArray(out, dims=rrs.dims[:-1] + ('wl_aeronet',),
                        coords={**{d: rrs[d] for d in rrs.dims[:-1] if d in rrs.coords}, 'wl_aeronet': wl_aer})


def band_depth(rrs):
    out = {}
    for b, l, r in DEPTHS:
        s = lambda x: rrs.sel(wl=x, method='nearest')
        if rrs.wl.max() < r or rrs.wl.min() > l:
            continue
        # skip when the band was removed (nearest band too far)
        if abs(float(rrs.wl.sel(wl=b, method='nearest')) - b) > 6:
            continue
        cont = s(l) + (s(r) - s(l)) * (b - l) / (r - l)
        out[f'{b:g}'] = (s(b) - cont).reset_coords(drop=True)
    return out


def load(version, aer):
    rows = []
    for mid in aer.matchup.values:
        f = OUT_DIR / 'runs' / version / f'{mid}.nc'
        if not f.exists():
            continue
        ds = xr.open_dataset(f).load()
        ds = ds.expand_dims(matchup=[mid])
        rows.append(ds)
    if not rows:
        return None
    # each product has its own band centres: put them on the grid of the first one (nearest, < 1 nm)
    ref = rows[0].wl.values
    out = []
    for ds in rows:
        ds = ds.drop_vars('Rrs_box', errors='ignore')
        wl_new = ref[np.argmin(np.abs(ref[None, :] - ds.wl.values[:, None]), axis=1)]
        spec = [v for v in ds.data_vars if 'wl' in ds[v].dims]
        sp = ds[spec].assign_coords(wl=wl_new)
        # only the spectral variables are regridded: a groupby on the whole dataset gives the scalars a wl
        # dimension
        sp = sp.groupby('wl').mean() if len(np.unique(wl_new)) < len(wl_new) else sp
        ds = xr.merge([ds.drop_vars(spec).drop_vars('wl', errors='ignore'), sp.reindex(wl=ref)])
        out.append(ds)
    return xr.concat(out, 'matchup', coords='minimal', compat='override', join='outer')


def stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(ok.sum())
    if n == 0:
        return dict(N=0)
    d = y - x
    # correlation and slope from 3 matchups (a single matchup still gives the differences)
    return dict(N=n, bias=float(np.median(d)), mad=float(np.median(np.abs(d))),
                rmsd=float(np.sqrt(np.mean(d ** 2))),
                mapd=float(100 * np.median(np.abs(d) / np.abs(x))),
                r=float(np.corrcoef(x, y)[0, 1]) if n >= 3 else np.nan,
                slope=float(np.polyfit(x, y, 1)[0]) if n >= 3 else np.nan)


def main(versions):
    CMP.mkdir(exist_ok=True)
    aer, n_bad = valid_aeronet(xr.open_dataset(OUT_DIR / 'aeronet_at_overpass.nc').load())
    print(f'{n_bad} AERONET-OC Rrs < {QC["rrs_min"]} sr-1 left out')
    wl_aer = aer.wl_aeronet.values
    data = {v: load(v, aer) for v in versions}
    data = {v: d for v, d in data.items() if d is not None}
    common = sorted(set.intersection(*[set(d.matchup.values) for d in data.values()]))
    n_all = len(common)
    common = latest_processing(common)
    print(f'{n_all} matchups processed by all versions, {n_all - len(common)} older DLR processings left out')
    # quality checks on the first version (same pixels for all)
    d0 = data[versions[0]].sel(matchup=common)
    cv = (d0.Rrs_std / d0.Rrs).sel(wl=560, method='nearest')
    ok = (d0.n_pix >= QC['n_pix_min']) & (np.abs(cv) <= QC['cv_max'])
    ok &= (aer.lag_min.sel(matchup=common) <= QC['lag_max'])
    keep = [m for m, k in zip(common, ok.values) if k]
    print(f'{len(common)} matchups, {len(keep)} pass the quality checks')
    pd.Series(keep).to_csv(CMP / 'matchups_qc.csv', index=False, header=['matchup'])
    a = aer.sel(matchup=keep)
    srows, prows, drows = [], [], []
    for v, d in data.items():
        d = d.sel(matchup=keep)
        sat = to_aeronet_bands(d.Rrs, wl_aer)
        xr.Dataset(dict(Rrs_aeronet=a.Rrs, Rrs_enmap=sat, aot_aeronet=a.aot550, aot_enmap=d.aot_ref,
                        tcwv_aeronet=a.tcwv, tcwv_enmap=d.tcwv, site=a.site)).to_netcdf(CMP / f'matchups_{v}.nc')
        for w in wl_aer:
            s = stats(a.Rrs.sel(wl_aeronet=w).values, sat.sel(wl_aeronet=w).values)
            srows.append(dict(version=v, wl=w, **s))
        for name, x, y in [('aot550', a.aot550, d.aot_ref), ('tcwv', a.tcwv, d.tcwv)]:
            prows.append(dict(version=v, param=name, **stats(x.values, y.values)))
        for where, spec in (('station', d.Rrs), ('dark', d.Rrs_dark)):
            for b, depth in band_depth(spec).items():
                x = depth.values[np.isfinite(depth.values)]
                drows.append(dict(version=v, where=where, band=b, N=x.size, median=np.median(x),
                                  mean_abs=np.mean(np.abs(x)), p25=np.percentile(x, 25), p75=np.percentile(x, 75)))
    srr = pd.DataFrame(srows)
    spa = pd.DataFrame(prows)
    sbd = pd.DataFrame(drows)
    srr.to_csv(CMP / 'stats_rrs.csv', index=False, float_format='%.6g')
    spa.to_csv(CMP / 'stats_params.csv', index=False, float_format='%.6g')
    sbd.to_csv(CMP / 'band_depth.csv', index=False, float_format='%.6g')
    pd.set_option('display.width', 200)
    for col in ('bias', 'rmsd', 'mapd'):
        print(f'\nRrs {col}')
        print(srr.pivot(index='wl', columns='version', values=col)[versions].round(5 if col != 'mapd' else 1)
              .to_string())
    print(srr.pivot(index='wl', columns='version', values='N')[versions].iloc[:, 0].to_string())
    print('\nparameters')
    print(spa.round(4).to_string())
    print('\nband depth (sr-1)')
    print(sbd.pivot_table(index=['where', 'band'], columns='version', values='median')[versions]
          .applymap(lambda x: f'{x:+.1e}').to_string())
    print(sbd.pivot_table(index=['where', 'band'], columns='version', values='mean_abs')[versions]
          .applymap(lambda x: f'{x:.1e}').to_string())


if __name__ == '__main__':
    main(sys.argv[1:] or ['v1.1.2', 'v1.2.0', 'v1.3dev'])
