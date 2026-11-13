'''
Coefficients of the photon path-length model of hgrs.gas_coupling over 364-2500 nm
(``hgrs/data/aux/gas_coupling_ppld_full.csv``).

The fit is that of radcalnet_oc (``scripts/sos_abs_coupling/fullrange.py``: SOS-ABS coupled simulations, sets
v2 (684-990 nm) and v3 (364-2500 nm outside 684-990 nm), CKD 5 cm-1 decomposed into exponential sums,
coefficients per window and term); this script only runs it and writes the table in the format of hGRS, with
its provenance. The simulations must exist (``make_luts.py v2`` and ``v3``).

Usage: python utils/fit_gas_coupling_full.py [radcalnet_oc_sos_abs_coupling_dir] [n_samples]
'''
import datetime
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'hgrs', 'data', 'aux', 'gas_coupling_ppld_full.csv')
SCRIPTS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/Dropbox/Dropbox/work/git/vrtc/radcalnet_oc/scripts/sos_abs_coupling')
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60000

sys.path.insert(0, SCRIPTS)
import fullrange as fr  # noqa: E402
import ppld  # noqa: E402


def git_commit(path):
    try:
        return subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def main():
    eff = fr.load_full()
    print('channels:', eff['T'].sizes['wl'], 'windows:',
          sorted(set(eff['window'].values.tolist()) - {''}), flush=True)
    a, rms = ppld.exponential_sum(eff['tg'])
    print(f'exponential sums: rms of Tg(m) median {float(rms.median()):.1e}, max {float(rms.max()):.1e}', flush=True)
    model = fr.fit_full(eff, a, n=N)
    table = fr.table_frame(model)
    commit = git_commit(SCRIPTS)
    with open(OUT, 'w') as f:
        f.write(
            f'# Fitted with radcalnet_oc (https://github.com/Tristanovsk/radcalnet_oc, Apache-2.0), commit {commit},\n'
            '# scripts/sos_abs_coupling/fullrange.py (fit_full) by hgrs utils/fit_gas_coupling_full.py, '
            f'{datetime.date.today()}; used by hgrs.gas_coupling.\n'
            '# Photon path-length model of the gaseous transmittance of the scattered light, per absorption window\n'
            '# and term (atm: path reflectance, w: water-leaving term, m = 1/mu0 + 1/muv; Ed: downwelling\n'
            '# irradiance, m = 1/mu0), 364-2500 nm:\n'
            '#   T = (1 + s2 eta tau)^(-1/s2), tau = -ln Tg(m) of the channel (Gamma distribution of the paths)\n'
            '#   eta = c0 + c1 (m - 2) + c2 (aot550 - 0.1) + c3 ln(U / 2 g cm-2)\n'
            '#   s2  = s0 + s1 (m - 2) + s2 (aot550 - 0.1) + s3 ln(U / 2 g cm-2), s2 >= 1e-4\n'
            '# U water vapour column (g cm-2); m, aot550 and U clipped to the ranges; rms: absolute transmittance.\n'
            '# Fitted on the SOS-ABS coupled simulations, sets v2 (684-990 nm) and v3 (364-2500 nm outside\n'
            '# 684-990 nm) of scripts/sos_abs_coupling/make_luts.py (CKD 5 cm-1 decomposed into exponential sums).\n')
        table.to_csv(f, index=False, float_format='%.6g')
    print(table.round(4).to_string())
    print('written', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
