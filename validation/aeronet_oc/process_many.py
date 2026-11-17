'''
Long-lived worker of ``run_versions.py``: process several matchups with one version of hGRS in the same Python
process, with :py:func:`process_one.main` (same results and files), so that the imports and the models loaded
once (e.g. the omnicloudmask networks, ~15 s) are shared by the matchups.

Protocol (line based): reads ``<matchup_index>\\t<log_file>`` on stdin, redirects the output of the matchup
(stdout, stderr, logging, C libraries) to ``log_file`` and answers ``<matchup_index>\\t<status>`` on stdout, with
status ``ok`` (result file), ``error`` (error file of a failed retrieval) or ``fail`` (exception out of the
retrieval, no file: retried by the next run). Ends at the end of stdin.

PRISMA crops are cached (``OUT_DIR/cache/prisma_crops``; ``HGRS_VAL_PRISMA_CACHE=0`` disables): the reader
georeferences the full scene (~3 min) before ``process_one.main`` crops it, so the cropped, georeferenced L1 is
saved by the first version and reloaded by the others. The key is the source of the ``hgrs`` package, the read
arguments, the station and the crop size: same input as without the cache, hence the same results.

Usage: python process_many.py version code_dir config|- half_size_px ncpu
'''
import hashlib
import logging
import os
import pickle
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import process_one  # noqa: E402
from common import OUT_DIR  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CACHE = os.environ.get('HGRS_VAL_PRISMA_CACHE', '1') != '0'
CACHE_DIR = OUT_DIR / 'cache' / 'prisma_crops'
logger = logging.getLogger('process_many')


def originals(code_dir, config):
    '''Import hGRS as :py:func:`process_one.main` does, and return the functions it patches, to restore them
    before each matchup (``main`` wraps the current reader: without the restore the crops would pile up).'''
    if config:
        os.environ['HGRS_CONFIG'] = str(config)
    sys.path.insert(0, str(code_dir))
    import hgrs
    if hasattr(hgrs, 'drivers'):
        from hgrs.drivers.enmap import EnmapDriver
        import hgrs.hgrs_process as hp
        saved = [(EnmapDriver, 'read', EnmapDriver.read)]
        try:                                        # patched by main as well (PRISMA scenes)
            from hgrs.drivers.prisma import PrismaDriver
            saved.append((PrismaDriver, 'read', PrismaDriver.read))
        except ImportError:
            pass
        if hasattr(hp, 'Pool'):
            saved.append((hp, 'Pool', hp.Pool))
    else:  # 1.1.x
        saved = [(hgrs.Driver, 'read_l1c_enmap', hgrs.Driver.read_l1c_enmap)]
    return saved


def source_hash(code_dir):
    '''hash of the Python source of the ``hgrs`` package of ``code_dir``'''
    h = hashlib.sha1()
    root = Path(code_dir) / 'hgrs'
    for f in sorted(root.rglob('*.py')):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def crop(ds, lon, lat, half):
    '''crop of ``process_one.main`` (same code): applied again there to the crop, it returns it unchanged'''
    from pyproj import Transformer
    crs = ds.rio.crs
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    ix = int(np.argmin(np.abs(ds.x.values - x)))
    iy = int(np.argmin(np.abs(ds.y.values - y)))
    if np.hypot(ds.x.values[ix] - x, ds.y.values[iy] - y) > 60:
        raise RuntimeError('station outside the image')
    out = ds.isel(x=slice(max(ix - half, 0), ix + half), y=slice(max(iy - half, 0), iy + half))
    return out.load() if hasattr(out, 'load') else out


def cache_prisma_read(read, src, mid, row, half):
    '''``PrismaDriver.read`` returning the crop of the matchup, from the cache if there (with the driver
    attributes set by the read, e.g. ``scene_metadata``), else read, cropped and saved'''
    def read_cached(self, img_path, **kwargs):
        key = hashlib.sha1(repr((src, [str(p) for p in img_path], sorted(kwargs.items()),
                                 float(row.lon), float(row.lat), half)).encode()).hexdigest()[:16]
        f = CACHE_DIR / f'{mid}__{key}.pkl'
        if f.exists():
            try:
                with open(f, 'rb') as fh:
                    ds, attrs = pickle.load(fh)
                self.__dict__.update(attrs)
                logger.info(f'cropped PRISMA image from the cache {f}')
                return ds
            except Exception:
                logger.exception(f'unreadable cache file {f}, read again')
        before = dict(self.__dict__)
        ds = crop(read(self, img_path, **kwargs), row.lon, row.lat, half)
        attrs = {k: v for k, v in self.__dict__.items() if k not in before or before[k] is not v}
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f'.{os.getpid()}.tmp')
            with open(tmp, 'wb') as fh:
                pickle.dump((ds, attrs), fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, f)
        except Exception:
            logger.exception(f'could not write the cache file {f}')
        return ds
    return read_cached


def main(version, code_dir, config, half, ncpu):
    # answers to the runner on a copy of stdout; any other output out of the matchups goes to stderr
    proto = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    saved_fds = os.dup(1), os.dup(2)
    patched = originals(code_dir, config)
    prisma_read = next((fn for obj, attr, fn in patched if obj.__name__ == 'PrismaDriver'), None) if CACHE else None
    src = source_hash(code_dir) if prisma_read else None
    matchups = pd.read_csv(OUT_DIR / 'matchups.csv')
    out_dir = OUT_DIR / 'runs' / version
    for line in sys.stdin:
        index, log = line.rstrip('\n').split('\t')
        index = int(index)
        r = matchups.iloc[index]
        mid = f'{r["product"]}__{r.site}'
        for obj, attr, fn in patched:
            if prisma_read is not None and fn is prisma_read:
                fn = cache_prisma_read(fn, src, mid, r, half)
            setattr(obj, attr, fn)
        sys.stdout.flush()
        sys.stderr.flush()
        with open(log, 'w') as fh:
            os.dup2(fh.fileno(), 1)
            os.dup2(fh.fileno(), 2)
            try:
                process_one.main(version, code_dir, index, config, half, ncpu)
            except BaseException:            # as an uncaught exception of process_one.py: no file
                traceback.print_exc()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_fds[0], 1)
                os.dup2(saved_fds[1], 2)
        status = 'ok' if (out_dir / f'{mid}.nc').exists() else \
            'error' if (out_dir / f'{mid}.error.json').exists() else 'fail'
        proto.write(f'{index}\t{status}\n')


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) != 5:
        sys.exit(__doc__)
    main(a[0], a[1], None if a[2] in ('', '-') else a[2], int(a[3]), int(a[4]))
