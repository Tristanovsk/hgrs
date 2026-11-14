'''
Database of hyperspectral images (EnMAP, PRISMA): storage layout, migration of existing folders, STAC catalogue,
GeoParquet index, views by site, search and ingest of new downloads.

Layout of a sensor folder (``root``, e.g. ``/data/satellite/enmap``)::

    <root>/l1c/2022/06/23/ENMAP01-____L1C-DT..._V010502_.../       one copy of each product (DLR, ASI),
    <root>/l2a-hgrs/v2.0.0/2022/06/23/<name>.nc                   by collection, [version,] acquisition date
    <root>/l2a-hgrs-snap/v1.1.0/..., <root>/l2w-acolite/...
    <root>/catalog/catalog.json                                   static STAC catalogue (layout of grstbx.stac and
    <root>/catalog/<collection>/collection.json, items.json         GRSdriver.stac: GRSdriver.stac.search reads it)
    <root>/catalog/<collection>/<id>/<id>.json
    <root>/catalog/products.parquet                               GeoParquet index (footprints), for search()
    <root>/catalog/sites.csv                                      point sites (name, lon, lat): sites of the images
    <root>/catalog/provenance.csv                                 former paths of each product (tags: folders)
    <root>/catalog/migration_log.csv                              operations of apply(), read by undo()
    <root>/sites/<site>/<collection>/<name> -> product            views (symlinks), rebuilt by build()
    <root>/projects/<name>/                                       working folders (validation, figures, tests)
    <root>/aux/                                                   documents (band tables, specifications)
    <root>/incoming/                                              new downloads, moved to the store by ingest()
    <root>/_trash/<former path>                                   redundant copies, to be deleted by hand
    <root>/legacy/<former path> -> new location                   former paths (symlinks), with a link
    <root>/<former top folder> -> legacy/<former top folder>        per former top folder (compatibility)

Collections: ``<sensor>-<folder>``, folders ``l1b``, ``l1c``, ``l2a-dlr`` (EnMAP, DLR), ``l1``, ``l2b``, ``l2c``,
``l2d`` (PRISMA, ASI), ``l2a-hgrs``, ``l2a-hgrs-snap`` (hGRS, NetCDF and SNAP/BEAM layouts) and ``l2w-acolite``.
The producer and version of the NetCDF products come from their attributes, not from their names.

Typical use::

    from hgrs import catalog
    plan = catalog.plan('/data/satellite/enmap')          # dry run: DataFrame of the operations, nothing moved
    catalog.apply(plan, '/data/satellite/enmap')          # moves (renames), links, log
    catalog.build('/data/satellite/enmap')                # STAC + GeoParquet + views by site
    catalog.search('/data/satellite/enmap', bbox=(12.4, 45.2, 12.6, 45.4), start='2023-01-01', level='L1C')
    catalog.ingest('/data/satellite/enmap')               # downloads of <root>/incoming
    catalog.undo('/data/satellite/enmap')                 # reverts the last apply()

or the command ``hgrs_catalog`` (``hgrs_catalog --help``). The search needs geopandas (``pip install hGRS[catalog]``).
'''
import argparse
import datetime
import json
import logging
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STAC_VERSION = '1.0.0'
STAC_EXTENSIONS = ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                   'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                   'https://stac-extensions.github.io/file/v2.1.0/schema.json']
MEDIA_JSON, MEDIA_GEOJSON = 'application/json', 'application/geo+json'

ENMAP = re.compile(r'ENMAP01-____(L1B|L1C|L2A|L2Asnap|L2A_hgrs)-DT(\d{10})_(\d{8}T\d{6})Z_(\d{3})_V(\d{6})_'
                   r'(\d{8}T\d{6})Z')
PRISMA = re.compile(r'PRS_(L1|L2[ABCD])_(?:STD_)?(?:OFFL_)?(\d{14})_(\d{14})_(\d{4})')
# folder of each (level, producer) in a sensor root
FOLDERS = {('L1B', 'dlr'): 'l1b', ('L1C', 'dlr'): 'l1c', ('L2A', 'dlr'): 'l2a-dlr', ('L1', 'asi'): 'l1',
           ('L2B', 'asi'): 'l2b', ('L2C', 'asi'): 'l2c', ('L2D', 'asi'): 'l2d', ('L2A', 'hgrs'): 'l2a-hgrs',
           ('L2A', 'hgrs-snap'): 'l2a-hgrs-snap', ('L2W', 'acolite'): 'l2w-acolite'}
VERSIONED = {'hgrs', 'hgrs-snap'}       # producers whose products are stored by version
TITLES = {'l1b': 'L1B (DLR)', 'l1c': 'L1C orthorectified TOA radiance (DLR)', 'l2a-dlr': 'L2A (DLR)',
          'l1': 'L1 TOA radiance (ASI)', 'l2b': 'L2B (ASI)', 'l2c': 'L2C surface reflectance (ASI)',
          'l2d': 'L2D geocoded surface reflectance (ASI)', 'l2a-hgrs': 'L2A Rrs by hGRS',
          'l2a-hgrs-snap': 'L2A Rrs by hGRS, SNAP/BEAM layout', 'l2w-acolite': 'L2W by ACOLITE'}
# folders of the layout (not scanned for products to migrate) and project folders moved as a whole
LAYOUT = {'catalog', 'sites', 'projects', 'aux', '_trash', 'legacy', 'incoming'}
RESERVED = set(FOLDERS.values()) | LAYOUT
PROJECT_UNITS = {'hgrs_validation'}
AUX_SUFFIXES = {'.pdf', '.xlsx', '.xls', '.docx'}
JUNK = re.compile(r'^(__MACOSX|\.~lock.*|\.ipynb_checkpoints|\.DS_Store|Thumbs\.db)$')
# folder names that say nothing about the content (not kept as tags)
GENERIC = re.compile(r'^(enmap_?l1c|enmap|prisma|l1|l1b|l1c|l2a|l2a_snap|l2a_test|l2c|l2d|l2w|acolite|archive\d*|fig|'
                     r'figs|figures|data)$', re.I)
# tags of download batches (not sites): no view by site
BATCH = re.compile(r'^(last|new|tmp|test|L\d[A-Z]?_[a-z]{3}\d{4}(_\w+)?|L2A_test)$', re.I)


@dataclass
class Product:
    path: str
    sensor: str
    level: str
    producer: str
    name: str                    # file or folder name
    product_id: str              # name without extension (zip: of the product inside)
    start: pd.Timestamp
    container: str               # dir, zip, he5, nc
    version: str = ''            # hGRS version, DLR processor version, ASI processor version
    source_id: str = ''          # L1 product of a derived product
    scene: str = ''              # same acquisition, whatever the processing
    processed: str = ''          # processing time of DLR products
    size: int = 0
    tags: list = field(default_factory=list)

    @property
    def folder(self):
        return FOLDERS[(self.level, self.producer)]

    def canonical(self, root):
        sub = [self.folder] + ([f'v{self.version}'] if self.producer in VERSIONED and self.version else [])
        name = self.product_id if self.container == 'zip' else self.name
        return Path(root, *sub, f'{self.start:%Y}', f'{self.start:%m}', f'{self.start:%d}', name)


# ---------------------------------------------------------------------------------------------------------------
# identification
def _attrs(path, keys):
    import h5py
    try:
        with h5py.File(path, 'r') as h:
            return {k: (h.attrs[k].decode() if isinstance(h.attrs[k], bytes) else h.attrs[k]) for k in keys
                    if k in h.attrs}
    except Exception as e:  # noqa: BLE001
        logger.warning('%s: %s', path, e)
        return None


def _from_name(name):
    '''sensor, level, start, scene, version, processing time and product id from a DLR / ASI name'''
    m = ENMAP.search(name)
    if m:
        return dict(sensor='enmap', level=m.group(1), start=pd.Timestamp(m.group(3)), version=m.group(5),
                    scene=f'DT{m.group(2)}_{m.group(3)}_{m.group(4)}', processed=m.group(6),
                    product_id=m.group(0))
    m = PRISMA.search(name)
    if m:
        return dict(sensor='prisma', level=m.group(1), start=pd.to_datetime(m.group(2), format='%Y%m%d%H%M%S'),
                    scene=f'{m.group(2)}_{m.group(3)}_{m.group(4)}', version='', processed='',
                    product_id=m.group(0))
    return None


def _walk_dirs(top):
    '''folders below ``top``, deepest first, without following the symlinks'''
    out = [Path(d) for d, _, _ in os.walk(top)]
    return sorted(out, key=lambda p: len(p.parts), reverse=True)


def _size(p):
    p = Path(p)
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def identify(path):
    '''Product of a folder (DLR), archive or file, or None'''
    p = Path(path)
    if p.is_dir():
        if not any(f.name.endswith('METADATA.XML') for f in p.iterdir()):
            return None
        n = _from_name(p.name)
        if n is None or n['sensor'] != 'enmap':
            return None
        return Product(str(p), 'enmap', n['level'], 'dlr', p.name, p.name, n['start'], 'dir', n['version'],
                       scene=n['scene'], processed=n['processed'], size=_size(p))
    suffix = p.suffix.lower()
    if suffix == '.zip':
        n = _from_name(p.name)
        if n is None:
            return None
        producer = 'dlr' if n['sensor'] == 'enmap' else 'asi'
        if n['level'] not in ('L1B', 'L1C', 'L2A', 'L1', 'L2B', 'L2C', 'L2D'):
            return None
        pid = n['product_id'] if n['sensor'] == 'enmap' else Path(p.name).stem
        return Product(str(p), n['sensor'], n['level'], producer, p.name, pid, n['start'], 'zip', n['version'],
                       scene=n['scene'], processed=n['processed'], size=p.stat().st_size)
    if suffix == '.he5':
        n = _from_name(p.name)
        if n is None or n['sensor'] != 'prisma':
            return None
        a = _attrs(p, ['Processor_Version']) or {}
        return Product(str(p), 'prisma', n['level'], 'asi', p.name, p.stem, n['start'], 'he5',
                       str(a.get('Processor_Version', '')), scene=n['scene'], size=p.stat().st_size)
    if suffix == '.nc':
        a = _attrs(p, ['generated_by', 'sensor', 'isodate', 'acolite_version', 'hgrs_version', 'L1C_product_name',
                       'metadata_profile'])
        if not a:
            return None
        if a.get('generated_by') == 'ACOLITE':
            sensor = 'enmap' if 'ENMAP' in str(a.get('sensor', '')).upper() else 'prisma'
            start = pd.Timestamp(a['isodate']).tz_localize(None) if 'isodate' in a else None
            if start is None:
                return None
            words = str(a.get('acolite_version', '')).split()
            version = re.sub(r'[^0-9A-Za-z.-]+', '', words[-1])[:20] if words else ''
            return Product(str(p), sensor, 'L2W', 'acolite', p.name, p.stem, start, 'nc', version,
                           size=p.stat().st_size)
        if 'hgrs_version' in a and 'L1C_product_name' in a:
            n = _from_name(str(a['L1C_product_name']))
            if n is None:
                return None
            producer = 'hgrs-snap' if a.get('metadata_profile') == 'beam' else 'hgrs'
            source = Path(str(a['L1C_product_name'])).name
            source = source[:-4] if source.endswith('.he5') else source
            return Product(str(p), n['sensor'], 'L2A', producer, p.name, p.stem, n['start'], 'nc',
                           str(a['hgrs_version']), source_id=source, scene=n['scene'], size=p.stat().st_size)
    return None


def _tags(rel_parent):
    return [t for t in Path(rel_parent).parts if not GENERIC.match(t) and not _from_name(t)]


# ---------------------------------------------------------------------------------------------------------------
# migration plan
def scan(root, include=None):
    '''Products and other files below ``root`` outside the layout: (list of Product, list of other paths)'''
    root = Path(root)
    products, other = [], []
    # the store folders are scanned too: products already at their place give no operation
    tops = [root / t for t in include] if include else sorted(p for p in root.iterdir() if p.name not in LAYOUT
                                                              and p.name != 'README.md')
    for top in tops:
        if top.is_symlink():
            continue
        if top.name in PROJECT_UNITS or top.is_file():
            other.append(top)
            continue
        for dirpath, dirs, files in os.walk(top):
            d = Path(dirpath)
            keep = []
            for s in sorted(dirs):
                p = d / s
                if p.is_symlink():
                    continue
                if JUNK.match(s):
                    other.append(p)
                    continue
                prod = identify(p)
                if prod is not None:
                    prod.tags = _tags(p.parent.relative_to(root))
                    products.append(prod)
                else:
                    keep.append(s)
            dirs[:] = keep
            for f in sorted(files):
                p = d / f
                if p.is_symlink():
                    continue
                prod = None if JUNK.match(f) else identify(p)
                if prod is not None:
                    prod.tags = _tags(p.parent.relative_to(root))
                    products.append(prod)
                else:
                    other.append(p)
    return products, other


def _keeper(group, root):
    '''copy kept among the copies of a product: not an archive, then the largest, then the one already at its
    place, then one outside a download batch, then the most recent'''
    return sorted(group, key=lambda p: (p.container != 'zip', p.size, Path(p.path) == p.canonical(root),
                                        not any(BATCH.match(t) for t in p.tags),
                                        Path(p.path).stat().st_mtime))[-1]


def plan(root, include=None):
    '''
    Operations to bring ``root`` to the layout (dry run, nothing is changed): DataFrame with columns ``op``
    (``move``, ``extract``, ``trash``, ``project``, ``aux``), ``src``, ``dst``, ``keeper`` (path where the former
    path will point), ``note`` and the product columns. ``include``: top folders to scan (default: all but the
    layout).
    '''
    root = Path(root)
    products, other = scan(root, include)
    rows = []
    groups = {}
    for p in products:
        groups.setdefault((p.folder, p.version if p.producer in VERSIONED else '', p.product_id), []).append(p)
    for key, group in groups.items():
        keep = _keeper(group, root)
        dst = keep.canonical(root)
        op = 'extract' if keep.container == 'zip' else 'move'
        rows.append(dict(op=op, src=keep.path, dst=str(dst), keeper=str(dst), note='', **_cols(keep, group)))
        for p in group:
            if p is keep:
                continue
            note = 'archive of the product' if p.container == 'zip' else (
                'same size' if p.size == keep.size else f'size differs: {p.size} vs {keep.size} (kept largest)')
            rows.append(dict(op='trash', src=p.path, dst=str(root / '_trash' / Path(p.path).relative_to(root)),
                             keeper=str(dst), note=note, **_cols(p, [p])))
    # folders without product are moved as a whole (the highest one below root, or below a folder with products)
    holders = {a for p in products for a in Path(p.path).parents}
    units = []
    for o in other:
        u = o
        while u.parent != root and u.parent not in holders:
            u = u.parent
        units.append(o if JUNK.match(o.name) and u == o else u)
    other = [u for u in dict.fromkeys(units)
             if not any(a in units for a in u.parents)]       # nothing inside a unit moved separately
    for o in other:
        rel = o.relative_to(root)
        if JUNK.match(o.name):
            op, dst = 'trash', root / '_trash' / rel
        elif o.name in PROJECT_UNITS and o.parent == root:
            op, dst = 'project', root / 'projects' / o.name
        elif o.is_file() and o.suffix.lower() in AUX_SUFFIXES:
            op, dst = 'aux', root / 'aux' / o.name
        elif o.parent == root:
            op, dst = 'project', root / 'projects' / 'misc' / o.name
        else:
            op, dst = 'project', root / 'projects' / rel
        rows.append(dict(op=op, src=str(o), dst=str(dst), keeper=str(dst), note='', size=_size(o)))
    df = pd.DataFrame(rows)
    if len(df):
        df = df[(df.src != df.dst) | (df.op == 'trash')].reset_index(drop=True)
        # aux: same file names from different folders
        dup = df.duplicated('dst', keep='first') & (df.op != 'trash')
        df.loc[dup, 'dst'] = [str(Path(d).with_name(f'{Path(d).stem}_{i}{Path(d).suffix}'))
                              for i, d in enumerate(df.loc[dup, 'dst'])]
    return df


def _cols(p, group):
    return dict(sensor=p.sensor, level=p.level, producer=p.producer, version=p.version, product_id=p.product_id,
                start=p.start, container=p.container, size=p.size, source_id=p.source_id, scene=p.scene,
                tags=';'.join(sorted({t for g in group for t in g.tags})))


def summary(df):
    '''sizes (GB) and counts of a plan per operation and collection'''
    df = df.fillna('')
    d = df.assign(GB=pd.to_numeric(df['size'], errors='coerce') / 1e9,
                  folder=[FOLDERS.get((lv, pr), '') for lv, pr in zip(df.level, df.producer)])
    return d.groupby(['op', 'folder'], dropna=False).agg(n=('src', 'size'), GB=('GB', 'sum')).round(2)


# ---------------------------------------------------------------------------------------------------------------
# apply / undo
def _busy(root):
    '''processes of the user with a file open or their working folder below ``root``'''
    root = str(Path(root).resolve())
    busy = set()
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            paths = [os.readlink(f'/proc/{pid}/cwd')]
            paths += [os.readlink(f'/proc/{pid}/fd/{fd}') for fd in os.listdir(f'/proc/{pid}/fd')]
        except OSError:
            continue
        if any(x == root or x.startswith(root + '/') for x in paths):
            busy.add(int(pid))
    busy.discard(os.getpid())
    return sorted(busy)


class _Log:
    def __init__(self, root):
        self.file = Path(root, 'catalog', 'migration_log.csv')
        self.file.parent.mkdir(parents=True, exist_ok=True)
        new = not self.file.exists()
        self.f = open(self.file, 'a')
        if new:
            self.f.write('time,op,src,dst\n')
        self.run = datetime.datetime.now().isoformat(timespec='seconds')
        self.f.write(f'{self.run},begin,,\n')

    def __call__(self, op, src, dst=''):
        self.f.write(f'{datetime.datetime.now().isoformat(timespec="seconds")},{op},"{src}","{dst}"\n')
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self('end', '')
        self.f.close()


def _move(src, dst, log, op='move'):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.rename(src, dst)            # same file system: a rename, nothing is copied
    log(op, src, dst)


def _extract(src, dst, root, log):
    '''unzip an archive of a product; the product folder (DLR: folder of the METADATA.XML) or file goes to dst'''
    inc = Path(root, 'incoming')
    inc.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=inc, prefix='.extract_'))
    try:
        with zipfile.ZipFile(src) as z:
            for m in z.namelist():
                if m.startswith('/') or '..' in Path(m).parts:
                    raise ValueError(f'{src}: unsafe member {m}')
            z.extractall(tmp)
        found = [Path(d) / n for d, dirs, files in os.walk(tmp) for n in dirs + files
                 if not n.lower().endswith('.zip') and identify(Path(d) / n) is not None]
        if len(found) != 1:
            raise RuntimeError(f'{src}: {len(found)} products in the archive')
        prod = found[0]
        dst = Path(dst).with_name(prod.name) if prod.is_file() else Path(dst)
        if dst.exists():
            raise FileExistsError(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(prod, dst)
        log('extract', src, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dst


def _check(df):
    '''every source exists, no destination exists, no destination twice'''
    errors = [f'missing source: {s}' for s in df.src if not (Path(s).exists() or Path(s).is_symlink())]
    errors += [f'destination exists: {d}' for d in df.dst if Path(d).exists() or Path(d).is_symlink()]
    errors += [f'destination twice: {d}' for d in df.dst[df.dst.duplicated()]]
    if errors:
        raise RuntimeError(f'{len(errors)} problems in the plan, nothing done:\n' + '\n'.join(errors[:20]))


def apply(df, root, legacy_links=True, force=False):
    '''
    Run a plan of :func:`plan`: products to the layout (archives extracted), redundant copies to ``_trash``,
    other files to ``projects`` and ``aux``, empty former folders removed, former paths kept as symlinks in
    ``legacy`` with a link per former top folder. Every operation is logged in ``catalog/migration_log.csv``
    (see :func:`undo`). Nothing is copied: the moves are renames within the file system. Refuses to run while a
    process uses ``root`` (``force`` to override), or if a source is missing or a destination exists.
    '''
    root = Path(root)
    df = df.fillna('')
    busy = _busy(root)
    if busy and not force:
        raise RuntimeError(f'processes using {root}: {busy} (wait for them, or force=True)')
    _check(df)
    log = _Log(root)
    actual = {}                 # planned destination -> actual location (extracted file names)
    zips = {}                   # archive -> its place in _trash
    try:
        order = {'extract': 0, 'move': 1, 'trash': 2, 'aux': 3, 'project': 4}
        for r in df.sort_values('op', key=lambda s: s.map(order), kind='stable').itertuples():
            if r.op == 'extract':
                actual[r.dst] = str(_extract(r.src, r.dst, root, log))
                zips[r.src] = root / '_trash' / Path(r.src).relative_to(root)
                _move(r.src, zips[r.src], log, 'trash')
            else:
                _move(r.src, r.dst, log, r.op)
        # tags of the products: their former folders
        prov = df[(df.product_id != '') & df.op.isin(['move', 'extract', 'trash'])] \
            if 'product_id' in df else df.iloc[:0]
        if len(prov):
            prov = pd.DataFrame(dict(path=[actual.get(k, k) for k in prov.keeper], former=prov.src,
                                     tags=prov.tags))
            f = root / 'catalog' / 'provenance.csv'
            prov.to_csv(f, mode='a', header=not f.exists(), index=False)
        # empty former folders
        tops = sorted({Path(s).relative_to(root).parts[0] for s in df.src})
        for top in tops:
            t = root / top
            if t.is_dir() and not t.is_symlink():
                for d in _walk_dirs(t):
                    if not any(d.iterdir()):
                        d.rmdir()
                        log('rmdir', d)
        if legacy_links:
            for r in df.itertuples():
                if r.op == 'extract':
                    target = zips[r.src]
                elif r.op == 'trash' and r.keeper != r.dst:       # redundant copy: the copy kept
                    target = Path(actual.get(r.keeper, r.keeper))
                else:
                    target = Path(actual.get(r.dst, r.dst))
                link = root / 'legacy' / Path(r.src).relative_to(root)
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.is_symlink():
                    link.symlink_to(os.path.relpath(target, link.parent))
                    log('link', link, target)
            for top in tops:
                t, old = root / top, root / 'legacy' / top
                if not old.is_dir():
                    continue
                if not t.exists() and not t.is_symlink():
                    t.symlink_to(Path('legacy') / top)
                    log('link', t, old)
                elif t.is_dir() and top in FOLDERS.values():
                    # former folder with the name of a store folder (e.g. l1c/loire): links in the store folder
                    for child in sorted(old.iterdir()):
                        if not (t / child.name).exists() and not (t / child.name).is_symlink():
                            (t / child.name).symlink_to(os.path.relpath(child, t))
                            log('link', t / child.name, child)
    finally:
        log.close()
    _write_readme(root)


def undo(root):
    '''Revert the last run of :func:`apply` from ``catalog/migration_log.csv``'''
    root = Path(root)
    f = root / 'catalog' / 'migration_log.csv'
    log = pd.read_csv(f, dtype=str, keep_default_na=False)
    begins = log.index[log.op == 'begin']
    if not len(begins):
        raise RuntimeError('nothing to undo')
    last = log.loc[begins[-1]:]
    for r in last.iloc[::-1].itertuples():
        if r.op == 'link':
            if Path(r.src).is_symlink():
                os.unlink(r.src)
        elif r.op in ('move', 'trash', 'project', 'aux'):
            Path(r.src).parent.mkdir(parents=True, exist_ok=True)
            os.rename(r.dst, r.src)
        elif r.op == 'extract':
            p = Path(r.dst)
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        elif r.op == 'rmdir':
            Path(r.src).mkdir(parents=True, exist_ok=True)
    # empty folders of the layout left by the undo
    for name in sorted(RESERVED - {'catalog'}):
        if (root / name).is_dir():
            for d in _walk_dirs(root / name):
                if not any(d.iterdir()):
                    d.rmdir()
    log.loc[:begins[-1] - 1].to_csv(f, index=False)
    prov = root / 'catalog' / 'provenance.csv'
    if prov.exists():
        p = pd.read_csv(prov, dtype=str, keep_default_na=False)
        p = p[[Path(x).exists() for x in p.path]]
        p.to_csv(prov, index=False) if len(p) else prov.unlink()


# ---------------------------------------------------------------------------------------------------------------
# catalogue
def _poly(pts):
    pts = [(round(float(x), 5), round(float(y), 5)) for x, y in pts]
    return {'type': 'Polygon', 'coordinates': [pts + pts[:1]]}


def footprint(p):
    '''GeoJSON footprint of a product (DLR metadata, ASI corners or geolocation, ACOLITE lat/lon) or None'''
    path = Path(p.path)
    try:
        if p.container == 'dir':
            xml = next(path.glob('*METADATA.XML'))
            poly = ET.parse(xml).getroot().find('.//spatialCoverage/boundingPolygon')
            pts = {q.find('frame').text: (float(q.find('longitude').text), float(q.find('latitude').text))
                   for q in poly.findall('point')}
            return _poly([pts[k] for k in ('upper_left', 'lower_left', 'lower_right', 'upper_right')])
        import h5py
        with h5py.File(path, 'r') as h:
            if p.container == 'he5':
                a = h.attrs
                if 'Product_ULcorner_lat' in a:
                    return _poly([(a[f'Product_{k}corner_long'], a[f'Product_{k}corner_lat'])
                                  for k in ('UL', 'LL', 'LR', 'UR')])
                names = []
                h.visit(lambda n: names.append(n) if n.endswith('Geolocation Fields/Latitude_VNIR') else None)
                if names:
                    lat, lon = h[names[0]], h[names[0].replace('Latitude', 'Longitude')]
                    ny, nx = lat.shape
                    c = [(0, 0), (ny - 1, 0), (ny - 1, nx - 1), (0, nx - 1)]
                    return _poly([(lon[i, j], lat[i, j]) for i, j in c])
            if p.producer == 'acolite' and 'lat' in h and 'lon' in h:
                lat, lon = h['lat'], h['lon']
                ny, nx = lat.shape
                c = [(0, 0), (ny - 1, 0), (ny - 1, nx - 1), (0, nx - 1)]
                return _poly([(lon[i, j], lat[i, j]) for i, j in c])
    except Exception as e:  # noqa: BLE001
        logger.warning('footprint of %s: %s', path, e)
    return None


def _cloud_cover(p):
    if p.container != 'dir':
        return None
    try:
        c = ET.parse(next(Path(p.path).glob('*METADATA.XML'))).getroot().find('.//cloudCover')
        return float(c.text) if c is not None else None
    except Exception:  # noqa: BLE001
        return None


def _inside(geom, lon, lat):
    '''point in polygon (ray casting)'''
    if not geom:
        return False
    ring = geom['coordinates'][0]
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def store_products(root):
    '''products of the layout of ``root`` (list of Product)'''
    root = Path(root)
    out = []
    for folder in sorted(set(FOLDERS.values())):
        base = root / folder
        for d, dirs, files in os.walk(base):
            depth = len(Path(d).relative_to(base).parts)
            keep = []
            for n in sorted(dirs):
                p = Path(d) / n
                prod = identify(p) if depth >= 3 else None      # YYYY/MM/DD/<product folder>
                if prod is not None and prod.folder == folder:
                    out.append(prod)
                elif not p.is_symlink():
                    keep.append(n)
            dirs[:] = keep
            for n in sorted(files):
                prod = identify(Path(d) / n) if depth >= 3 else None
                if prod is not None and prod.folder == folder:
                    out.append(prod)
    return out


def build(root, workers=8):
    '''
    Write the STAC catalogue, the GeoParquet index (``catalog/products.parquet``, needs geopandas) and the views
    by site of the products of the layout. Sites of a product: its former folders (``provenance.csv``), except
    the download batches, and the point sites of ``catalog/sites.csv`` inside its footprint.
    Returns the index (DataFrame).
    '''
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root)
    cat = root / 'catalog'
    cat.mkdir(exist_ok=True)
    products = store_products(root)
    with ThreadPoolExecutor(workers) as pool:
        geoms = list(pool.map(footprint, products))
        clouds = list(pool.map(_cloud_cover, products))
    prov = pd.read_csv(cat / 'provenance.csv', dtype=str, keep_default_na=False) \
        if (cat / 'provenance.csv').exists() else pd.DataFrame(columns=['path', 'former', 'tags'])
    tags = prov.groupby('path').tags.agg(lambda s: sorted({t for x in s for t in x.split(';') if t}))
    sites = pd.read_csv(cat / 'sites.csv') if (cat / 'sites.csv').exists() else pd.DataFrame(
        columns=['name', 'lon', 'lat'])
    rows = []
    by_id = {}
    for p, g, cc in zip(products, geoms, clouds):
        t = tags.get(p.path, [])
        s = sorted({x for x in t if not BATCH.match(x)}
                   | {r.name for r in sites.itertuples() if _inside(g, r.lon, r.lat)})
        rows.append(dict(id=p.product_id, collection=f'{p.sensor}-{p.folder}', sensor=p.sensor, level=p.level,
                         producer=p.producer, version=p.version, datetime=p.start, scene=p.scene,
                         processed=p.processed, source_id=p.source_id, container=p.container, path=p.path,
                         size=p.size, cloud_cover=cc, sites=';'.join(s), tags=';'.join(t), geometry=g))
        if p.producer in ('dlr', 'asi'):
            by_id[p.product_id] = rows[-1]
    df = pd.DataFrame(rows)
    if not len(df):
        logger.warning('no product in %s', root)
        return df
    # footprint of the derived products: that of their source
    for r in rows:
        if r['geometry'] is None and r['source_id'] in by_id:
            r['geometry'] = by_id[r['source_id']]['geometry']
            r['sites'] = r['sites'] or by_id[r['source_id']]['sites']
    df = pd.DataFrame(rows)
    # latest processing of each scene (DLR, ASI) and latest hGRS version of each source product
    key = np.where(df.producer.isin(list(VERSIONED)), df.source_id + df.producer,
                   np.where(df.scene != '', df.scene + df.collection, df.id))
    rank = df.processed.where(df.processed != '', df.version.map(_version_key))
    df['latest'] = rank == pd.Series(rank).groupby(key).transform('max')
    _write_stac(root, df)
    _write_index(root, df)
    _write_views(root, df)
    _write_readme(root)
    return df


def _version_key(v):
    return '.'.join(f'{int(x):04d}' if x.isdigit() else x for x in str(v).replace('V', '').split('.'))


def _bbox(g):
    if not g:
        return None
    x, y = zip(*g['coordinates'][0])
    return [min(x), min(y), max(x), max(y)]


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(content, indent=1, default=str))
    os.replace(tmp, path)


def _write_stac(root, df):
    cat = Path(root, 'catalog')
    children = []
    ids = set(df.id[df.producer.isin(['dlr', 'asi'])])
    for coll, d in df.groupby('collection'):
        items = []
        folder = coll.split('-', 1)[1]
        for r in d.sort_values('datetime').itertuples():
            path = Path(r.path)
            assets = {'data': {'href': str(path), 'roles': ['data'], 'title': TITLES.get(folder, folder),
                               'file:size': int(r.size)}}
            if r.container == 'dir':
                for kind, suffix, roles in (('metadata', 'METADATA.XML', ['metadata']),
                                            ('quicklook_vnir', 'QL_VNIR.TIF', ['overview']),
                                            ('quicklook_swir', 'QL_SWIR.TIF', ['overview'])):
                    f = next(path.glob(f'*{suffix}'), None)
                    if f:
                        assets[kind] = {'href': str(f), 'roles': roles}
            props = {'datetime': pd.Timestamp(r.datetime).isoformat() + 'Z', 'platform': r.sensor,
                     'instruments': ['hsi'], 'processing:level': r.level,
                     'processing:software': {r.producer: r.version} if r.version else None,
                     'eo:cloud_cover': None if pd.isna(r.cloud_cover) else r.cloud_cover,
                     'hsi:scene': r.scene or None, 'hsi:latest': bool(r.latest),
                     'hsi:sites': r.sites.split(';') if r.sites else None,
                     'hsi:tags': r.tags.split(';') if r.tags else None}
            links = [{'rel': 'root', 'href': '../../catalog.json', 'type': MEDIA_JSON},
                     {'rel': 'parent', 'href': '../collection.json', 'type': MEDIA_JSON},
                     {'rel': 'collection', 'href': '../collection.json', 'type': MEDIA_JSON}]
            if r.source_id in ids:
                src = df[(df.id == r.source_id) & df.producer.isin(['dlr', 'asi'])].iloc[0]
                links.append({'rel': 'derived_from', 'type': MEDIA_GEOJSON,
                              'href': f'../../{src.collection}/{src.id}/{src.id}.json'})
            item = {'type': 'Feature', 'stac_version': STAC_VERSION, 'stac_extensions': STAC_EXTENSIONS,
                    'id': r.id, 'collection': coll, 'geometry': r.geometry, 'bbox': _bbox(r.geometry),
                    'properties': {k: v for k, v in props.items() if v is not None}, 'assets': assets,
                    'links': links}
            _write_json(cat / coll / r.id / f'{r.id}.json', item)
            items.append(item)
        _write_json(cat / coll / 'items.json', {'type': 'FeatureCollection', 'features': items})
        boxes = [b for b in (_bbox(g) for g in d.geometry) if b]
        bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                max(b[3] for b in boxes)] if boxes else [-180, -90, 180, 90]
        _write_json(cat / coll / 'collection.json', {
            'type': 'Collection', 'stac_version': STAC_VERSION, 'id': coll,
            'title': f'{d.sensor.iloc[0].upper()} {TITLES.get(folder, folder)}', 'description': TITLES.get(folder),
            'license': 'proprietary',
            'extent': {'spatial': {'bbox': [bbox]},
                       'temporal': {'interval': [[pd.Timestamp(d.datetime.min()).isoformat() + 'Z',
                                                  pd.Timestamp(d.datetime.max()).isoformat() + 'Z']]}},
            'summaries': {'processing:level': sorted(set(d.level)), 'platform': sorted(set(d.sensor))},
            'links': [{'rel': 'root', 'href': '../catalog.json', 'type': MEDIA_JSON},
                      {'rel': 'parent', 'href': '../catalog.json', 'type': MEDIA_JSON},
                      {'rel': 'items', 'href': './items.json', 'type': MEDIA_GEOJSON}]
            + [{'rel': 'item', 'href': f'./{i}/{i}.json', 'type': MEDIA_GEOJSON} for i in d.id]})
        children.append({'rel': 'child', 'href': f'./{coll}/collection.json', 'type': MEDIA_JSON,
                         'title': TITLES.get(folder, folder)})
    _write_json(cat / 'catalog.json', {
        'type': 'Catalog', 'stac_version': STAC_VERSION, 'id': f'{Path(root).name}-products',
        'title': f'{Path(root).name} hyperspectral images', 'description': f'Products stored in {root}',
        'links': [{'rel': 'root', 'href': './catalog.json', 'type': MEDIA_JSON}] + children})


def _write_index(root, df):
    f = Path(root, 'catalog', 'products.parquet')
    try:
        import geopandas as gpd
        from shapely.geometry import shape
    except ImportError:
        logger.warning('geopandas not installed: %s not written (pip install hGRS[catalog])', f)
        return
    g = gpd.GeoDataFrame(df.drop(columns='geometry'),
                         geometry=[shape(x) if x else None for x in df.geometry], crs='EPSG:4326')
    g.to_parquet(f)


def _write_views(root, df):
    root = Path(root)
    views = root / 'sites'
    if views.exists():
        for d, dirs, files in os.walk(views, topdown=False):
            for n in dirs + files:
                p = Path(d) / n
                if p.is_symlink():
                    p.unlink()
                elif p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
    for r in df.itertuples():
        for s in filter(None, r.sites.split(';')):
            folder = r.collection.split('-', 1)[1]
            sub = [folder] + ([f'v{r.version}'] if r.producer in VERSIONED and r.version else [])
            link = views.joinpath(s, *sub, Path(r.path).name)
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.is_symlink():
                link.symlink_to(os.path.relpath(r.path, link.parent))


def _write_readme(root):
    Path(root, 'README.md').write_text(f'''# {Path(root).name}: hyperspectral images

Managed with `hgrs.catalog` (`hgrs_catalog --help`): do not file products by hand, put new downloads in
`incoming/` and run `hgrs_catalog ingest {root}`.

- `l1c/`, `l1/`, `l2c/`, ...: one copy of each product, `<collection>/[v<version>/]YYYY/MM/DD/<product>`
- `l2a-hgrs/v<version>/`, `l2a-hgrs-snap/`: hGRS products (NetCDF, SNAP/BEAM layout); `l2w-acolite/`: ACOLITE
- `catalog/`: STAC catalogue (`catalog.json`), GeoParquet index (`products.parquet`), `sites.csv` (point sites),
  `provenance.csv` (former paths), `migration_log.csv`
- `sites/<site>/`: views by site (symlinks), rebuilt by `hgrs_catalog build`
- `projects/`: working folders; `aux/`: documents; `incoming/`: downloads to ingest
- `_trash/`: redundant copies set aside by the migration, to delete by hand
- `legacy/`: former paths (symlinks to the new locations); the former top folders link there. Remove them
  (`rm -r legacy` and the links of the top folder) once no script uses the former paths.
''')


# ---------------------------------------------------------------------------------------------------------------
# search and ingest
def search(root, bbox=None, point=None, start=None, end=None, collection=None, level=None, producer=None,
           site=None, version=None, max_cloud_cover=None, latest=True):
    '''
    Products of the index of ``root`` (GeoDataFrame, needs geopandas), selected by area (``bbox`` (W, S, E, N)
    or ``point`` (lon, lat)), period, collection, level, producer, site, version and cloud cover; ``latest``: only
    the latest processing of each scene (and latest hGRS version of each source product).
    '''
    import geopandas as gpd
    from shapely.geometry import Point, box
    df = gpd.read_parquet(Path(root, 'catalog', 'products.parquet'))
    if bbox is not None:
        df = df[df.intersects(box(*bbox))]
    if point is not None:
        df = df[df.contains(Point(*point))]
    if start is not None:
        df = df[df.datetime >= pd.Timestamp(start)]
    if end is not None:
        # a date alone includes the whole day
        stop = pd.Timestamp(end) + (pd.Timedelta(days=1) if len(str(end)) <= 10 else pd.Timedelta(0))
        df = df[(df.datetime < stop) if len(str(end)) <= 10 else (df.datetime <= stop)]
    for col, val in (('collection', collection), ('level', level), ('producer', producer), ('version', version)):
        if val is not None:
            df = df[df[col].isin([val] if isinstance(val, str) else val)]
    if site is not None:
        df = df[df.sites.str.split(';').map(lambda s: site in s)]
    if max_cloud_cover is not None:
        df = df[df.cloud_cover.isna() | (df.cloud_cover <= max_cloud_cover)]
    if latest:
        df = df[df.latest]
    return df.sort_values('datetime')


def ingest(root, paths=None, site=None):
    '''
    Move new downloads (default: everything in ``<root>/incoming``) to the layout, archives extracted (the
    archive goes to ``_trash``), then rebuild the catalogue. ``site``: tag of the products (view by site).
    Products already in the store go to ``_trash``. Returns the plan run.
    '''
    root = Path(root)
    inc = root / 'incoming'
    if paths is None:
        inc.mkdir(exist_ok=True)
        paths = [p for p in inc.iterdir() if not p.name.startswith('.')]
    products = [x for x in (identify(p) for p in paths) if x is not None]
    existing = {(p.folder, p.product_id) for p in store_products(root)}
    rows = []
    for p in products:
        p.tags = [site] if site else []
        dst = p.canonical(root)
        if (p.folder, p.product_id) in existing:
            rows.append(dict(op='trash', src=p.path, dst=str(root / '_trash' / 'incoming' / p.name),
                             keeper=str(dst), note='already in the store', **_cols(p, [p])))
        else:
            rows.append(dict(op='extract' if p.container == 'zip' else 'move', src=p.path, dst=str(dst),
                             keeper=str(dst), note='', **_cols(p, [p])))
    df = pd.DataFrame(rows)
    if len(df):
        apply(df, root, legacy_links=False, force=True)
        build(root)
    return df


def seed_sites(root, matchups_csv):
    '''``catalog/sites.csv`` from the stations of a matchup table (columns site, lon, lat)'''
    m = pd.read_csv(matchups_csv)
    s = m.groupby('site')[['lon', 'lat']].first().reset_index().rename(columns={'site': 'name'})
    f = Path(root, 'catalog', 'sites.csv')
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        s = pd.concat([pd.read_csv(f), s]).drop_duplicates('name')
    s.to_csv(f, index=False)
    return s


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('plan', help='dry run: write the operations to a CSV and print a summary')
    s.add_argument('root')
    s.add_argument('-o', '--out', default=None, help='CSV of the plan (default: <root>/catalog/plan.csv)')
    s = sub.add_parser('apply', help='run a plan (CSV of plan)')
    s.add_argument('root')
    s.add_argument('plan_csv', nargs='?')
    s.add_argument('--no-legacy-links', action='store_true')
    s.add_argument('--force', action='store_true', help='run even if processes use the folder')
    s = sub.add_parser('undo', help='revert the last apply')
    s.add_argument('root')
    s = sub.add_parser('build', help='STAC catalogue, GeoParquet index and views by site')
    s.add_argument('root')
    s = sub.add_parser('ingest', help='move the downloads of <root>/incoming (or PATHS) to the store')
    s.add_argument('root')
    s.add_argument('paths', nargs='*')
    s.add_argument('--site')
    s = sub.add_parser('search', help='search the index')
    s.add_argument('root')
    s.add_argument('--bbox', type=float, nargs=4)
    s.add_argument('--point', type=float, nargs=2)
    s.add_argument('--start')
    s.add_argument('--end')
    s.add_argument('--level')
    s.add_argument('--collection')
    s.add_argument('--site')
    s.add_argument('--all', action='store_true', help='all processings, not only the latest')
    s = sub.add_parser('seed-sites', help='catalog/sites.csv from a matchup table (site, lon, lat)')
    s.add_argument('root')
    s.add_argument('matchups_csv')
    o = a.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    pd.set_option('display.width', 200, 'display.max_rows', 200, 'display.max_colwidth', 100)
    if o.cmd == 'plan':
        df = plan(o.root)
        out = Path(o.out or Path(o.root, 'catalog', 'plan.csv'))
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(summary(df).to_string())
        print(f'\n{len(df)} operations written to {out}')
    elif o.cmd == 'apply':
        f = o.plan_csv or Path(o.root, 'catalog', 'plan.csv')
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        apply(df, o.root, legacy_links=not o.no_legacy_links, force=o.force)
    elif o.cmd == 'undo':
        undo(o.root)
    elif o.cmd == 'build':
        df = build(o.root)
        print(df.groupby('collection').agg(n=('id', 'size'), GB=('size', lambda s: round(s.sum() / 1e9, 1))))
    elif o.cmd == 'ingest':
        print(ingest(o.root, o.paths or None, o.site))
    elif o.cmd == 'search':
        df = search(o.root, bbox=o.bbox, point=o.point, start=o.start, end=o.end, level=o.level,
                    collection=o.collection, site=o.site, latest=not o.all)
        print(df[['datetime', 'collection', 'version', 'sites', 'path']].to_string(index=False))
    elif o.cmd == 'seed-sites':
        print(seed_sites(o.root, o.matchups_csv))


if __name__ == '__main__':
    main()
