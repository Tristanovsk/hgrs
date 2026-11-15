'''
Long-lived worker of ``run_versions.py``: process several matchups with one version of hGRS in the same Python
process, with :py:func:`process_one.main` (same results and files), so that the imports and the models loaded
once (e.g. the omnicloudmask networks, ~15 s) are shared by the matchups.

Protocol (line based): reads ``<matchup_index>\\t<log_file>`` on stdin, redirects the output of the matchup
(stdout, stderr, logging, C libraries) to ``log_file`` and answers ``<matchup_index>\\t<status>`` on stdout, with
status ``ok`` (result file), ``error`` (error file of a failed retrieval) or ``fail`` (exception out of the
retrieval, no file: retried by the next run). Ends at the end of stdin.

Usage: python process_many.py version code_dir config|- half_size_px ncpu
'''
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import process_one  # noqa: E402
from common import OUT_DIR  # noqa: E402

import pandas as pd  # noqa: E402


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


def main(version, code_dir, config, half, ncpu):
    # answers to the runner on a copy of stdout; any other output out of the matchups goes to stderr
    proto = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    saved_fds = os.dup(1), os.dup(2)
    patched = originals(code_dir, config)
    matchups = pd.read_csv(OUT_DIR / 'matchups.csv')
    out_dir = OUT_DIR / 'runs' / version
    for line in sys.stdin:
        index, log = line.rstrip('\n').split('\t')
        index = int(index)
        r = matchups.iloc[index]
        mid = f'{r["product"]}__{r.site}'
        for obj, attr, fn in patched:
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
