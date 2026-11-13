'''
Common settings and readers of the AERONET-OC validation of hGRS (EnMAP).

Paths (environment variables override the defaults):

* ``HGRS_VAL_ENMAP``: EnMAP L1C products, one folder per site or delivery (searched recursively);
* ``HGRS_VAL_AERONET``: root of the AERONET-OC version 3 database (``aeronet_oc_db``: raw files
  ``<site>_OCv3.lev20|lev15`` and the products in ``db/``, updated with ``aeronet-oc-db update``);
* ``HGRS_VAL_CAMS``: CAMS files, ``cams_forecast_YYYY-MM.nc`` or ``YYYY/MM/DD/YYYY-MM-DD-*.nc``;
* ``HGRS_VAL_OUT``: output folder of the validation (matchup tables, simulations, L2A products).
'''
import glob
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ENMAP_DIR = Path(os.environ.get('HGRS_VAL_ENMAP', '/data/satellite/enmap/EnMAP_L1C'))
AERONET_DIR = Path(os.environ.get('HGRS_VAL_AERONET', '/DATA/AERONET/OCv3'))
CAMS_DIR = Path(os.environ.get('HGRS_VAL_CAMS', '/data/cams/world'))
OUT_DIR = Path(os.environ.get('HGRS_VAL_OUT', '/data/satellite/enmap/hgrs_validation'))


# ------------------------------------------------------------------------------------------------
# EnMAP scenes
# ------------------------------------------------------------------------------------------------
def enmap_scenes(root=ENMAP_DIR):
    '''Table of the EnMAP L1C products found under ``root``: product id, folder, L1 processor version,
    start time and footprint polygon (lon, lat).'''
    rows = []
    for xml in sorted(glob.glob(str(root / '**' / '*-METADATA.XML'), recursive=True)):
        product = os.path.dirname(xml)
        name = os.path.basename(product)
        if not name.startswith('ENMAP01-____L1C'):
            continue
        try:
            tree = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        start = tree.find('.//startTime')
        poly = tree.find('.//boundingPolygon')
        if start is None or poly is None:
            continue
        pts = [(float(p.find('longitude').text), float(p.find('latitude').text)) for p in poly.findall('point')]
        version = re.search(r'_V(\d{6})_', name)
        rows.append(dict(product=name, path=product, version=version.group(1) if version else '',
                         time=pd.Timestamp(start.text.rstrip('Z')), footprint=pts))
    df = pd.DataFrame(rows)
    # the same datatake is delivered several times (sites, reprocessings): keep every copy, they are
    # identified by the product id
    return df.drop_duplicates('product').reset_index(drop=True)


def inside(lon, lat, polygon):
    '''point in polygon (ray casting)'''
    x, y = np.asarray(polygon).T
    n = len(x)
    ok = False
    j = n - 1
    for i in range(n):
        if ((y[i] > lat) != (y[j] > lat)) and (lon < (x[j] - x[i]) * (lat - y[i]) / (y[j] - y[i]) + x[i]):
            ok = not ok
        j = i
    return ok


# ------------------------------------------------------------------------------------------------
# AERONET-OC
# ------------------------------------------------------------------------------------------------
def aeronet_db(root=AERONET_DIR):
    '''AERONET-OC database (:class:`aeronet_oc_db.Database`)'''
    from aeronet_oc_db import Database
    return Database(root)


def aeronet_times(site, level, root=AERONET_DIR):
    '''times (UTC) of the records of a site'''
    with aeronet_db(root).open(site, level) as ds:
        return pd.DatetimeIndex(ds.time.values)


def aeronet_sites(root=AERONET_DIR):
    '''Sites of the AERONET-OC database: name, longitude, latitude and, for the levels 2.0 and 1.5, the
    netCDF file of the site (``None`` if the site has no data at that level)'''
    db = aeronet_db(root)
    cat = db.catalog
    if cat.empty:
        raise FileNotFoundError(f'empty AERONET-OC catalog in {db.dir}: run "aeronet-oc-db build"')
    out = []
    for site, g in cat.groupby('site', sort=False):
        levels = set(g.level)
        out.append(dict(site=site, lon=float(g.longitude.iloc[0]), lat=float(g.latitude.iloc[0]),
                        **{lev: str(db.netcdf_path(site, lev)) if lev in levels else None
                           for lev in ('lev20', 'lev15')}))
    return pd.DataFrame(out)


def cams_file(time, root=CAMS_DIR):
    '''CAMS file of a date: daily file if present, else the monthly forecast file'''
    t = pd.Timestamp(time)
    daily = sorted(glob.glob(str(root / f'{t:%Y}' / f'{t:%m}' / f'{t:%d}' / '*.nc')))
    if daily:
        return daily[0]
    monthly = root / f'cams_forecast_{t:%Y-%m}.nc'
    return str(monthly) if monthly.exists() else None
