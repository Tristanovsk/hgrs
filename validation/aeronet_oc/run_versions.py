'''
Process all the matchups of ``matchups.csv`` with several versions of hGRS (``process_one.py``, one matchup and
version at a time per slot, ``n_parallel`` slots). Matchups already processed (result or error file) are skipped.

Versions: ``name=code_dir[:config.yml]``; the default versions are the code exported in ``OUT_DIR/code/<name>``.

Each slot runs its matchups in a long-lived worker (``process_many.py``, one per version), so that the imports
and the models (omnicloudmask, ~15 s) are loaded once per worker rather than once per matchup; a worker is
replaced after HGRS_VAL_JOBS_PER_WORKER matchups (default 25; 0: one process per matchup, as before) or when it
dies. Its stderr is in ``OUT_DIR/runs/<name>/logs/worker*.log``.

Each worker (or process) runs in its own transient systemd scope with a memory cap, so one exceeding it is
OOM-killed alone (its pool workers included) instead of pushing the session into systemd-oomd territory, which
kills the largest app (e.g. PyCharm); the matchup in progress is then reported and retried by the next run.
Caps: HGRS_VAL_MEM_MAX (hard, OOM-kill), HGRS_VAL_SWAP_MAX, HGRS_VAL_MEM_HIGH (soft throttle, off by default:
a job above it stalls for ever instead of being killed); HGRS_VAL_MEM_MAX=0 disables.

Usage: python run_versions.py [n_parallel] [name[=code_dir[:config]] ...]
'''
import itertools
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import OUT_DIR

HERE = Path(__file__).resolve().parent
PY = sys.executable
# CPUs of the pool of the retrievals of each process (HGRS_VAL_NCPU)
NCPU = os.environ.get('HGRS_VAL_NCPU', '8')
HALF = '150'
# matchups per long-lived worker (0: one process_one.py per matchup)
JOBS_PER_WORKER = int(os.environ.get('HGRS_VAL_JOBS_PER_WORKER', '25'))
# memory caps of each job (process_one.py + its pool), systemd size syntax
MEM_HIGH = os.environ.get('HGRS_VAL_MEM_HIGH', '')
MEM_MAX = os.environ.get('HGRS_VAL_MEM_MAX', '12G')
SWAP_MAX = os.environ.get('HGRS_VAL_SWAP_MAX', '512M')


def capped(cmd, unit):
    '''Wrap ``cmd`` in a memory-capped transient user scope (no-op without systemd-run or if disabled).'''
    if MEM_MAX == '0' or not shutil.which('systemd-run'):
        return cmd
    props = [f'MemoryMax={MEM_MAX}', f'MemorySwapMax={SWAP_MAX}', 'OOMPolicy=kill']
    if MEM_HIGH:
        props.append(f'MemoryHigh={MEM_HIGH}')
    return ['systemd-run', '--user', '--scope', '--quiet', '--collect', f'--unit={unit}',
            *(a for p in props for a in ('-p', p)), '--', *cmd]


def unit_name(*parts):
    return '-'.join(['hgrs', *map(str, parts), str(os.getpid())]).replace('_', '-')


def parse(spec):
    name, _, rest = spec.partition('=')
    code, _, config = (rest or str(OUT_DIR / 'code' / name)).partition(':')
    return name, code, config or '-'


class Worker:
    '''``process_many.py`` of one version, in its memory-capped scope'''
    ids = itertools.count()

    def __init__(self, name, code, config):
        self.key = name, code, config
        self.n_jobs = 0
        k = next(Worker.ids)
        log_dir = OUT_DIR / 'runs' / name / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.stderr = open(log_dir / f'worker{k}_{os.getpid()}.log', 'w')
        cmd = [PY, str(HERE / 'process_many.py'), name, code, config, HALF, NCPU]
        self.proc = subprocess.Popen(capped(cmd, unit_name(name, f'w{k}')), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)

    def run(self, i, log):
        '''status of matchup ``i``: ok, error (error file), fail (no file) or ``exit <rc>`` (worker died)'''
        self.n_jobs += 1
        try:
            self.proc.stdin.write(f'{i}\t{log}\n')
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except BrokenPipeError:
            line = ''
        if not line:
            return f'exit {self.proc.wait()}'
        return line.rstrip('\n').split('\t')[1]

    @property
    def alive(self):
        return self.proc.poll() is None

    def close(self):
        if self.alive:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.stderr.close()


_local = threading.local()
_workers = []
_lock = threading.Lock()


def run(job):
    name, code, config, i, mid = job
    log = OUT_DIR / 'runs' / name / 'logs' / f'{mid}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    if JOBS_PER_WORKER <= 0:
        with open(log, 'w') as fh:
            cmd = [PY, str(HERE / 'process_one.py'), name, code, str(i), config, HALF, NCPU]
            r = subprocess.run(capped(cmd, unit_name(name, i)), stdout=fh, stderr=subprocess.STDOUT)
        return name, mid, 'ok' if r.returncode == 0 else f'exit {r.returncode}'
    w = getattr(_local, 'worker', None)
    if w is not None and (w.key != (name, code, config) or w.n_jobs >= JOBS_PER_WORKER or not w.alive):
        w.close()
        w = None
    if w is None:
        w = _local.worker = Worker(name, code, config)
        with _lock:
            _workers.append(w)
    return name, mid, w.run(i, log)


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
    print(f'{len(jobs)} jobs, {n_parallel} in parallel, '
          f'{f"{JOBS_PER_WORKER} matchups per worker" if JOBS_PER_WORKER > 0 else "one process per matchup"}, '
          f'memory cap per worker: '
          f'{"none" if MEM_MAX == "0" else f"max {MEM_MAX} / swap {SWAP_MAX}" + (f" / high {MEM_HIGH}" if MEM_HIGH else "")}',
          flush=True)
    try:
        with ThreadPoolExecutor(n_parallel) as ex:
            for name, mid, status in ex.map(run, jobs):
                if status in ('exit -9', 'exit 137'):
                    status += f' (killed: likely over the {MEM_MAX} memory cap, rerun with a higher HGRS_VAL_MEM_MAX)'
                print(name, mid, status, flush=True)
    finally:
        for w in _workers:
            w.close()


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(n, sys.argv[2:] or ['v1.1.2', 'v1.2.0', 'v1.3dev'])
