'''
Common settings and readers of the AERONET-OC validation of hGRS (EnMAP).

Paths (environment variables override the defaults):

* ``HGRS_VAL_ENMAP``: EnMAP L1C products, one folder per site or delivery (searched recursively);
* ``HGRS_VAL_AERONET``: AERONET-OC version 3 files (``<site>_OCv3.lev20`` and ``.lev15``);
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
def read_aeronet_oc(site_file):
    '''AERONET-OC v3 file -> DataFrame (radcalnet_oc reader), time index in UTC, columns
    (field, type, wavelength)'''
    import radcalnet_oc.aeronet_oc as ao
    return ao.Aeronet().read_aeronet_ocv3(str(site_file))


def aeronet_sites(root=AERONET_DIR):
    '''Sites of the AERONET-OC files: name, longitude, latitude, files of level 2.0 and 1.5'''
    rows = {}
    for f in sorted(glob.glob(str(root / '*_OCv3.lev*'))):
        site, level = re.match(r'(.*)_OCv3\.(lev\d\d)$', os.path.basename(f)).groups()
        rows.setdefault(site, {})[level] = f
    out = []
    for site, files in rows.items():
        lon = lat = np.nan
        lines = []
        for level in ('lev20', 'lev15'):          # first file with data (some files are empty)
            if files.get(level):
                with open(files[level], encoding='latin_1') as fh:
                    lines = fh.readlines()
                if len(lines) > 8:
                    break
                files[level] = None
        if len(lines) <= 8:
            continue
        header = lines[7].strip().split(',')
        try:
            ilat, ilon = header.index('Site_Latitude(Degrees)'), header.index('Site_Longitude(Degrees)')
            for line in lines[8:]:
                v = line.strip().split(',')
                if len(v) > max(ilat, ilon):
                    lat, lon = float(v[ilat]), float(v[ilon])
                    break
        except ValueError:
            pass
        out.append(dict(site=site, lon=lon, lat=lat, lev20=files.get('lev20'), lev15=files.get('lev15')))
    return pd.DataFrame(out)


def cams_file(time, root=CAMS_DIR):
    '''CAMS file of a date: daily file if present, else the monthly forecast file'''
    t = pd.Timestamp(time)
    daily = sorted(glob.glob(str(root / f'{t:%Y}' / f'{t:%m}' / f'{t:%d}' / '*.nc')))
    if daily:
        return daily[0]
    monthly = root / f'cams_forecast_{t:%Y-%m}.nc'
    return str(monthly) if monthly.exists() else None
