'''Tests of hgrs.catalog on a synthetic tree of EnMAP / PRISMA products'''
import os
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from hgrs import catalog

L1C = 'ENMAP01-____L1C-DT0000001199_20220623T104344Z_015_V010502_20250308T051735Z'
L1C_B = 'ENMAP01-____L1C-DT0000001199_20220623T104344Z_015_V010502_20250401T000000Z'   # reprocessing
L1C_Z = 'ENMAP01-____L1C-DT0000086069_20240723T111414Z_008_V010502_20250417T115758Z'   # archive only
PRS = 'PRS_L2C_STD_20220307111720_20220307111724_0001'
XML = '''<?xml version="1.0"?><level_X><base><spatialCoverage><boundingPolygon>
<point><frame>upper_left</frame><latitude>45.6</latitude><longitude>12.2</longitude></point>
<point><frame>lower_left</frame><latitude>45.3</latitude><longitude>12.1</longitude></point>
<point><frame>lower_right</frame><latitude>45.2</latitude><longitude>12.6</longitude></point>
<point><frame>upper_right</frame><latitude>45.5</latitude><longitude>12.7</longitude></point>
<point><frame>center</frame><latitude>45.4</latitude><longitude>12.4</longitude></point>
</boundingPolygon></spatialCoverage></base><cloudCover>12</cloudCover></level_X>'''


def l1c(folder, name, payload=b'x' * 100):
    d = Path(folder, name)
    d.mkdir(parents=True)
    (d / f'{name}-METADATA.XML').write_text(XML)
    (d / f'{name}-SPECTRAL_IMAGE.TIF').write_bytes(payload)
    (d / f'{name}-QL_VNIR.TIF').write_bytes(b'q')
    return d


def nc(path, **attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as h:
        h['x'] = np.arange(3)
        for k, v in attrs.items():
            h.attrs[k] = v
    return path


def he5(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as h:
        for k, (lon, lat) in dict(UL=(5.0, 43.6), LL=(5.0, 43.3), LR=(5.4, 43.3), UR=(5.4, 43.6)).items():
            h.attrs[f'Product_{k}corner_long'] = lon
            h.attrs[f'Product_{k}corner_lat'] = lat
        h.attrs['Processor_Version'] = b'02.05'


def tree(root):
    '''listing of a tree: relative path -> (kind, size or link target)'''
    out = {}
    for d, dirs, files in os.walk(root):
        for n in dirs + files:
            p = Path(d, n)
            rel = str(p.relative_to(root))
            out[rel] = ('link', os.readlink(p)) if p.is_symlink() else (
                ('dir', 0) if p.is_dir() else ('file', p.stat().st_size))
    return out


@pytest.fixture
def root(tmp_path):
    r = tmp_path / 'enmap'
    l1c(r / 'EnMAP_L1C' / 'AAOT', L1C)
    l1c(r / 'EnMAP_L1C' / 'last', L1C)                          # same product in a batch folder
    l1c(r / 'EnMAP_L1C' / 'AAOT', L1C_B)
    src = l1c(r / 'tmp_zip', L1C_Z)
    (r / 'l1c' / 'loire').mkdir(parents=True)
    with zipfile.ZipFile(r / 'l1c' / 'loire' / f'{L1C_Z}-20250429T125758Z-001.zip', 'w') as z:
        for f in src.iterdir():
            z.write(f, f'{L1C_Z}-20250429T125758Z-001/{L1C_Z}/{f.name}')
    for f in src.iterdir():
        f.unlink()
    src.rmdir()
    (r / 'tmp_zip').rmdir()
    nc(r / 'L2A' / 'AAOT' / f'{L1C}.nc', hgrs_version='1.1.0', L1C_product_name=L1C)
    nc(r / 'L2A' / 'last' / f'{L1C}.nc', hgrs_version='1.1.0', L1C_product_name=L1C)
    nc(r / 'l2a_snap' / 'AAOT' / f'{L1C}.nc', hgrs_version='1.1.0', L1C_product_name=L1C,
       metadata_profile='beam')
    nc(r / 'L2A' / 'AAOT' / f'{L1C}_v2.nc', hgrs_version='2.0.0', L1C_product_name=L1C)
    he5(r / 'berre' / 'L1C' / f'{PRS}.he5')
    (r / 'fig').mkdir()
    (r / 'fig' / 'rgb.png').write_bytes(b'png')
    (r / 'bands.xlsx').write_bytes(b'xl')
    (r / 'hgrs_validation' / 'runs').mkdir(parents=True)
    (r / 'hgrs_validation' / 'runs' / f'{L1C}__AAOT.nc').write_bytes(b'run')
    (r / '.~lock.bands.xlsx#').write_bytes(b'')
    return r


def test_plan_apply_build_search_undo(root):
    before = tree(root)
    df = catalog.plan(root)
    ops = df.groupby('op').size().to_dict()
    assert ops == {'move': 6, 'extract': 1, 'trash': 3, 'project': 2, 'aux': 1}
    # nothing changed by the plan
    assert tree(root) == before

    catalog.apply(df, root, force=True)
    d = root / 'l1c' / '2022' / '06' / '23'
    assert (d / L1C).is_dir() and (d / L1C_B).is_dir()
    assert (root / 'l1c' / '2024' / '07' / '23' / L1C_Z / f'{L1C_Z}-METADATA.XML').exists()
    assert (root / 'l2a-hgrs' / 'v1.1.0' / '2022' / '06' / '23' / f'{L1C}.nc').exists()
    assert (root / 'l2a-hgrs' / 'v2.0.0' / '2022' / '06' / '23' / f'{L1C}_v2.nc').exists()
    assert (root / 'l2a-hgrs-snap' / 'v1.1.0' / '2022' / '06' / '23' / f'{L1C}.nc').exists()
    assert (root / 'l2c' / '2022' / '03' / '07' / f'{PRS}.he5').exists()
    assert (root / '_trash' / 'EnMAP_L1C' / 'last' / L1C).is_dir()
    assert (root / 'projects' / 'hgrs_validation' / 'runs').is_dir()
    assert (root / 'aux' / 'bands.xlsx').exists()
    # former paths still work
    assert (root / 'EnMAP_L1C' / 'AAOT' / L1C / f'{L1C}-METADATA.XML').exists()
    assert (root / 'EnMAP_L1C' / 'last' / L1C).resolve() == (d / L1C).resolve()
    assert (root / 'hgrs_validation' / 'runs').is_dir()
    assert (root / 'l1c' / 'loire' / f'{L1C_Z}-20250429T125758Z-001.zip').is_file()       # in _trash

    catalog.seed_sites(root, _sites(root))
    idx = catalog.build(root)
    assert set(idx.collection) == {'enmap-l1c', 'enmap-l2a-hgrs', 'enmap-l2a-hgrs-snap', 'prisma-l2c'}
    i = idx.set_index(['collection', 'id'])
    assert i.latest[('enmap-l1c', L1C_B)] and not i.latest[('enmap-l1c', L1C)]
    assert i.latest[('enmap-l2a-hgrs', f'{L1C}_v2')] and not i.latest[('enmap-l2a-hgrs', L1C)]
    assert i.loc[('enmap-l1c', L1C)].sites == 'AAOT' and i.loc[('enmap-l1c', L1C)].cloud_cover == 12
    assert i.loc[('enmap-l1c', L1C_Z)].sites == 'AAOT;loire'
    assert i.loc[('enmap-l2a-hgrs', f'{L1C}_v2')].geometry is not None       # footprint of the source
    assert i.loc[('prisma-l2c', PRS)].geometry is not None
    assert (root / 'catalog' / 'enmap-l1c' / L1C / f'{L1C}.json').exists()
    assert (root / 'sites' / 'AAOT' / 'l1c' / L1C).resolve() == (d / L1C).resolve()
    assert not (root / 'sites' / 'last').exists()                             # batch: no view

    pytest.importorskip('geopandas')
    found = catalog.search(root, point=(12.4, 45.4), level='L1C')
    assert list(found.id) == [L1C_B, L1C_Z]
    assert len(catalog.search(root, point=(12.4, 45.4), level='L1C', latest=False)) == 3
    assert len(catalog.search(root, site='AAOT', start='2022-06-23', end='2022-06-23', level='L1C')) == 1
    assert len(catalog.search(root, bbox=(0, 0, 1, 1))) == 0

    # undo restores the tree (the catalogue and the views of build() stay)
    for p in ('sites', 'README.md'):
        q = root / p
        catalog.shutil.rmtree(q) if q.is_dir() else q.unlink()
    catalog.undo(root)
    after = {k: v for k, v in tree(root).items() if not k.startswith('catalog')}
    assert after == before


def _sites(root):
    f = root.parent / 'matchups.csv'
    pd.DataFrame(dict(site=['AAOT', 'Far'], lon=[12.4, 100.], lat=[45.4, 0.])).to_csv(f, index=False)
    return f


def test_ingest(root):
    catalog.apply(catalog.plan(root), root, force=True)
    inc = root / 'incoming'
    l1c(inc, 'ENMAP01-____L1C-DT0000200000_20260101T100000Z_001_V010506_20260102T000000Z')
    l1c(inc / 'dup', L1C)
    df = catalog.ingest(root, [inc / 'ENMAP01-____L1C-DT0000200000_20260101T100000Z_001_V010506_20260102T000000Z',
                               inc / 'dup' / L1C], site='Thau')
    assert sorted(df.op) == ['move', 'trash']
    assert (root / 'sites' / 'Thau' / 'l1c').is_dir()
    assert (root / 'l1c' / '2026' / '01' / '01').is_dir()
