'''
TOA reflectance of the EnMAP L1C products at the AERONET-OC stations (``matchups.csv``): median and
standard deviation of a ``WINDOW`` x ``WINDOW`` pixel box centred on the station, geometry at the
station. Writes ``OUT_DIR/enmap_at_stations.nc`` (dimensions ``matchup``, ``wl``).

Usage: python extract_enmap.py [n_workers]
'''
import sys
import warnings
from types import SimpleNamespace
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

from common import OUT_DIR

warnings.filterwarnings('ignore')
WINDOW = 5


def extract(row):
    from pyproj import Transformer
    from hgrs.drivers.enmap import EnmapDriver
    mid = f'{row.product}__{row.site}'
    try:
        driver = EnmapDriver()
        ds = driver.read(row.path)
        crs = ds.rio.crs
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(row.lon, row.lat)
        ix = int(np.argmin(np.abs(ds.x.values - x)))
        iy = int(np.argmin(np.abs(ds.y.values - y)))
        dist = np.hypot(ds.x.values[ix] - x, ds.y.values[iy] - y)
        h = WINDOW // 2
        if not (h <= ix < ds.sizes['x'] - h and h <= iy < ds.sizes['y'] - h) or dist > 60:
            return mid, None, 'station outside the image'
        box = ds.isel(x=slice(ix - h, ix + h + 1), y=slice(iy - h, iy + h + 1))[['Rtoa', 'sza', 'vza', 'raa']].load()
        rtoa = box.Rtoa.load()
        out = xr.Dataset(dict(
            Rtoa=rtoa.median(['x', 'y']), Rtoa_std=rtoa.std(['x', 'y']),
            n_valid=rtoa.isel(wl=0).notnull().sum(),
            sza=box.sza.median(), vza=box.vza.median(), raa=box.raa.median(),
            fwhm=ds.fwhm.load()))
        out = out.reset_coords(drop=True)
        out.attrs.update(product=row.product, site=row.site)
        return mid, out, ''
    except Exception as e:
        return mid, None, repr(e)[:200]


def main(n_workers=4):
    m = pd.read_csv(OUT_DIR / 'matchups.csv', parse_dates=['time', 'aeronet_time'])
    out_file = OUT_DIR / 'enmap_at_stations.nc'
    rows = [SimpleNamespace(**r) for r in m.to_dict('records')]   # picklable rows
    results, errors = {}, {}
    with Pool(n_workers) as pool:
        for mid, ds, err in pool.imap_unordered(extract, rows):
            if ds is None:
                errors[mid] = err
                print('skip', mid, err, flush=True)
            else:
                results[mid] = ds
                print('ok', mid, flush=True)
    ids = sorted(results)
    # common wavelength grid: each product has its own band centres (within ~0.1 nm): keep them per matchup
    wl_ref = results[ids[0]].wl.values
    def on_ref(ds):
        ds = ds.assign_coords(band=('wl', np.arange(ds.sizes['wl']))).swap_dims(wl='band')
        return ds.rename(wl='wl_band')
    data = xr.concat([on_ref(results[i]) for i in ids], 'matchup', coords='different', compat='equals', join='outer')
    data = data.assign_coords(matchup=ids)
    data.attrs['window'] = WINDOW
    data.to_netcdf(out_file)
    pd.Series(errors, name='error').to_csv(OUT_DIR / 'extract_enmap_errors.csv')
    print(f'{len(ids)} matchups extracted, {len(errors)} skipped -> {out_file}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
