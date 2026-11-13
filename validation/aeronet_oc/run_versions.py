'''
Process all the matchups of ``matchups.csv`` with several versions of hGRS (``process_one.py``, one process per
matchup and version, ``n_parallel`` at a time). Matchups already processed (result or error file) are skipped.

Versions: ``name=code_dir[:config.yml]``; the default versions are the code exported in ``OUT_DIR/code/<name>``.

Usage: python run_versions.py [n_parallel] [name[=code_dir[:config]] ...]
'''
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import OUT_DIR

HERE = Path(__file__).resolve().parent
PY = sys.executable
# CPUs of the pool of the retrievals of each process (HGRS_VAL_NCPU)
NCPU = os.environ.get('HGRS_VAL_NCPU', '8')


def parse(spec):
    name, _, rest = spec.partition('=')
    code, _, config = (rest or str(OUT_DIR / 'code' / name)).partition(':')
    return name, code, config or '-'


def run(job):
    name, code, config, i, mid = job
    log = OUT_DIR / 'runs' / name / 'logs' / f'{mid}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as fh:
        r = subprocess.run([PY, str(HERE / 'process_one.py'), name, code, str(i), config, '150', NCPU],
                           stdout=fh, stderr=subprocess.STDOUT)
    return name, mid, r.returncode


def main(n_parallel, specs):
    m = pd.read_csv(OUT_DIR / 'matchups.csv')
    jobs = []
    for spec in specs:
        name, code, config = parse(spec)
        for i, r in m.iterrows():
            if pd.isna(r.level):
                continue
            mid = f'{r["product"]}__{r.site}'
            d = OUT_DIR / 'runs' / name
            if (d / f'{mid}.nc').exists() or (d / f'{mid}.error.json').exists():
                continue
            jobs.append((name, code, config, i, mid))
    print(f'{len(jobs)} jobs', flush=True)
    with ThreadPoolExecutor(n_parallel) as ex:
        for name, mid, rc in ex.map(run, jobs):
            print(name, mid, 'ok' if rc == 0 else f'exit {rc}', flush=True)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(n, sys.argv[2:] or ['v1.1.2', 'v1.2.0', 'v1.3dev'])
