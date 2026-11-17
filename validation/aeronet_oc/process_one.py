'''
Process one matchup of ``matchups.csv`` with one version of hGRS, on a crop of the EnMAP L1C (or PRISMA L1 +
L2C, ``path`` = ``<l1>;<l2c>``) image centred on the AERONET-OC station, and write the results at the station to
``OUT_DIR/runs/<version>/<product>__<site>.nc``:

* ``Rrs`` (wl): median of the water pixels of a ``BOX`` x ``BOX`` pixel box centred on the station,
  ``Rrs_std`` and ``n_pix``; ``Rrs_box`` (wl, yb, xb), the box itself;
* retrieved parameters of the mega-pixel of the station (``aot_ref``, ``tcwv``, ``brdfg``, their std),
  and crop means (``aot_ref_crop``, ``tcwv_crop``);
* ``Rrs_dark`` (wl): median spectrum of the 5 % darkest water pixels of the crop (at 865 nm), used for the
  spectral residuals in the absorption bands.

The code of the version is imported from ``code_dir`` (exported with ``git archive``), with the settings of the
optional YAML file ``config`` (``HGRS_CONFIG``).

Usage: python process_one.py version code_dir matchup_index [config.yml] [half_size_px] [ncpu]
'''
import functools
import logging
import os
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT_DIR, image_path  # noqa: E402

BOX = 15          # box of the station (pixels, 450 m)
CORE = 5          # core of the box for the median (pixels, 150 m)


def main(version, code_dir, index, config=None, half=150, ncpu=8):
    if config:
        os.environ['HGRS_CONFIG'] = str(config)
    sys.path.insert(0, str(code_dir))
    import hgrs
    from pyproj import Transformer
    assert Path(hgrs.__file__).resolve().is_relative_to(Path(code_dir).resolve()), hgrs.__file__
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

    m = pd.read_csv(OUT_DIR / 'matchups.csv', parse_dates=['time'])
    row = m.iloc[index]
    mid = f'{row["product"]}__{row.site}'
    out_dir = OUT_DIR / 'runs' / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{mid}.nc'
    state = {}

    def crop(ds):
        crs = ds.rio.crs
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(row.lon, row.lat)
        ix = int(np.argmin(np.abs(ds.x.values - x)))
        iy = int(np.argmin(np.abs(ds.y.values - y)))
        if np.hypot(ds.x.values[ix] - x, ds.y.values[iy] - y) > 60:
            raise RuntimeError('station outside the image')
        state.update(crs=crs, x=float(ds.x.values[ix]), y=float(ds.y.values[iy]))
        out = ds.isel(x=slice(max(ix - half, 0), ix + half), y=slice(max(iy - half, 0), iy + half))
        return out.load() if hasattr(out, 'load') else out

    # crop the L1C image as soon as it is read, and limit the pool of the retrievals
    if hasattr(hgrs, 'drivers'):
        from hgrs.drivers.enmap import EnmapDriver
        drivers = [EnmapDriver]
        try:
            from hgrs.drivers.prisma import PrismaDriver
            drivers.append(PrismaDriver)
        except ImportError:
            pass
        for cls in drivers:
            def read_cropped(self, *args, _read=cls.read, **kwargs):
                return crop(_read(self, *args, **kwargs))
            cls.read = functools.wraps(cls.read)(read_cropped)
        import hgrs.hgrs_process as hp
        if hasattr(hp, 'Pool'):
            hp.Pool = functools.partial(Pool, ncpu)
    else:  # 1.1.x
        read = hgrs.Driver.read_l1c_enmap

        @functools.wraps(read)
        def read_cropped(self, *args, **kwargs):
            return crop(read(self, *args, **kwargs))
        hgrs.Driver.read_l1c_enmap = read_cropped

    t0 = time.time()
    try:
        proc = hgrs.Process()
        proc.execute(image_path(row.path), row.cams)
        l2 = proc.l2_prod
        if l2 is None:
            raise RuntimeError('no L2 product')
    except Exception as e:
        traceback.print_exc()
        pd.Series(dict(matchup=mid, error=repr(e)[:300])).to_json(out_dir / f'{mid}.error.json')
        return
    elapsed = time.time() - t0

    ix = int(np.argmin(np.abs(l2.x.values - state['x'])))
    iy = int(np.argmin(np.abs(l2.y.values - state['y'])))
    h = BOX // 2
    box = l2.Rrs.isel(x=slice(ix - h, ix + h + 1), y=slice(iy - h, iy + h + 1))
    water = l2.water_mask.isel(x=slice(ix - h, ix + h + 1), y=slice(iy - h, iy + h + 1)) \
        if 'water_mask' in l2 else xr.ones_like(box.isel(wl=0), dtype=bool)
    box = box.where(water)
    c = CORE // 2
    core = box.isel(x=slice(h - c, h + c + 1), y=slice(h - c, h + c + 1))
    out = xr.Dataset(dict(Rrs=core.median(['x', 'y']), Rrs_std=core.std(['x', 'y']),
                          n_pix=core.isel(wl=0).notnull().sum(),
                          Rrs_box=box.rename(x='xb', y='yb').drop_vars(['xb', 'yb'], errors='ignore')))
    out = out.reset_coords(drop=True)
    # parameters of the mega-pixel of the station
    for v in ('aot_ref', 'aot_ref_std', 'tcwv', 'tcwv_std', 'brdfg', 'brdfg_std', 'aot_ref_smoothed'):
        if v in l2:
            f = l2[v]
            out[v] = f.sel(xc=state['x'], yc=state['y'], method='nearest').reset_coords(drop=True) \
                if 'xc' in f.dims else f.sel(x=state['x'], y=state['y'], method='nearest').reset_coords(drop=True)
            out[v + '_crop'] = f.mean().reset_coords(drop=True)
    # darkest water pixels of the crop (865 nm)
    rrs = l2.Rrs
    if 'water_mask' in l2:
        rrs = rrs.where(l2.water_mask)
    r865 = rrs.sel(wl=865, method='nearest')
    thr = float(r865.quantile(0.05))
    dark = rrs.where(r865 <= thr)
    out['Rrs_dark'] = dark.median(['x', 'y']).reset_coords(drop=True)
    out['n_dark'] = dark.isel(wl=0).notnull().sum().reset_coords(drop=True)
    for v in ('pressure', 'to3c', 'tno2c', 'tch4c', 'aerosol_model'):
        if v in l2 and l2[v].size == 1:
            out[v] = l2[v].reset_coords(drop=True)
    out.attrs.update(matchup=mid, product=row['product'], site=row.site, version=version,
                     hgrs_version=getattr(hgrs, '__version__', ''), code_dir=str(code_dir),
                     config=str(config or ''), half_size_px=half, elapsed_s=elapsed,
                     station_x=state['x'], station_y=state['y'])
    for k, v in list(out.attrs.items()):
        if not isinstance(v, (str, int, float)):
            out.attrs[k] = str(v)
    out.to_netcdf(out_file)
    print(f'{mid}: done in {elapsed:.0f} s -> {out_file}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit(__doc__)
    main(a[0], a[1], int(a[2]), a[3] if len(a) > 3 and a[3] not in ('', '-') else None,
         int(a[4]) if len(a) > 4 else 150, int(a[5]) if len(a) > 5 else 8)
