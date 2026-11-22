'''
Image gallery of the validation sites (appendix of the article): for one matchup per site and sensor, the crop
around the station processed with hGRS at full resolution, keeping the true-colour TOA reflectance, the
true-colour remote-sensing reflectance (BOA), the sunglint reflectance and the water mask.

The matchup of each site is the one of ``compare/matchups_qc.csv`` (quality checks passed) with the largest
sunglint reflectance of the station mega-pixel in the run of ``version`` (``brdfg``), to show the correction.

Usage (``HGRS_VAL_SENSOR``, ``HGRS_VAL_OUT`` as for ``run_versions.py``)::

    python gallery.py select <version>                      # gallery/selection.csv
    python gallery.py one <code_dir> <config> <index>       # gallery/<matchup>.nc (one row of selection.csv)

Run ``one`` in a memory-capped scope per matchup (see ``CLAUDE.local.md``).
'''
import functools
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import OUT_DIR, image_path

warnings.filterwarnings('ignore')
GALLERY = OUT_DIR / 'gallery'
RGB = (655., 560., 490.)


def select(version):
    m = pd.read_csv(OUT_DIR / 'matchups.csv')
    m['matchup'] = m['product'] + '__' + m.site
    qc_file = OUT_DIR / 'compare' / 'matchups_qc.csv'
    if qc_file.exists():
        m = m[m.matchup.isin(pd.read_csv(qc_file).matchup)]
    glint = {}
    for mid in m.matchup:
        f = OUT_DIR / 'runs' / version / f'{mid}.nc'
        if f.exists():
            with xr.open_dataset(f) as r:
                glint[mid] = float(r.brdfg) if 'brdfg' in r else np.nan
    m['brdfg'] = m.matchup.map(glint)
    m = m.dropna(subset=['brdfg']).sort_values('brdfg', ascending=False).drop_duplicates('site')
    GALLERY.mkdir(parents=True, exist_ok=True)
    m.sort_values('site').to_csv(GALLERY / 'selection.csv', index=False)
    print(len(m), 'sites ->', GALLERY / 'selection.csv')


def one(code_dir, config, index, half=150, ncpu=4):
    if config:
        os.environ['HGRS_CONFIG'] = str(config)
    sys.path.insert(0, str(code_dir))
    import hgrs
    from pyproj import Transformer
    import hgrs.hgrs_process as hp
    from hgrs.drivers.enmap import EnmapDriver
    from hgrs.drivers.prisma import PrismaDriver
    row = pd.read_csv(GALLERY / 'selection.csv').iloc[int(index)]
    state = {}

    def crop(ds):
        x, y = Transformer.from_crs(4326, ds.rio.crs, always_xy=True).transform(row.lon, row.lat)
        ix, iy = int(np.argmin(np.abs(ds.x.values - x))), int(np.argmin(np.abs(ds.y.values - y)))
        state.update(x=float(ds.x.values[ix]), y=float(ds.y.values[iy]))
        out = ds.isel(x=slice(max(ix - half, 0), ix + half), y=slice(max(iy - half, 0), iy + half))
        return out.load() if hasattr(out, 'load') else out

    for cls in (EnmapDriver, PrismaDriver):
        def read_cropped(self, *args, _read=cls.read, **kwargs):
            return crop(_read(self, *args, **kwargs))
        cls.read = functools.wraps(cls.read)(read_cropped)
    if hasattr(hp, 'Pool'):
        hp.Pool = functools.partial(Pool, ncpu)
    proc = hgrs.Process()
    proc.execute(image_path(row.path), row.cams)
    l1, l2 = proc.l1_prod, proc.l2_prod
    rtoa = l1.Rtoa.transpose('wl', 'y', 'x')
    out = xr.Dataset(dict(
        Rtoa_rgb=(('rgb', 'y', 'x'), np.stack([rtoa.sel(wl=w, method='nearest').values for w in RGB])),
        Rrs_rgb=(('rgb', 'y', 'x'), np.stack([l2.Rrs.sel(wl=w, method='nearest').values for w in RGB])),
        sunglint=(('y', 'x'), l2.brdfg_full.values),
        water_mask=(('y', 'x'), l2.water_mask.values.astype('uint8'))),
        coords=dict(rgb=list(RGB), y=l2.y.values, x=l2.x.values))
    out.attrs.update(matchup=f'{row["product"]}__{row.site}', site=row.site, time=str(row.time),
                     station_x=state['x'], station_y=state['y'], hgrs_version=hgrs.__version__,
                     config=str(config), aot_ref=float(np.nanmean(l2.aot_ref_smoothed.values)))
    for v in ('Rtoa_rgb', 'Rrs_rgb', 'sunglint'):
        out[v] = out[v].astype('float32')
    out.to_netcdf(GALLERY / f'{out.attrs["matchup"]}.nc')
    print('written', out.attrs['matchup'])


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        select(sys.argv[2])
    else:
        one(*sys.argv[2:5])
