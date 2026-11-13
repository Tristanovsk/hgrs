r'''
Screening of water vapour retrieval models on the EnMAP TOA spectra of the stations (``closure/closure.nc``),
against the water vapour column of AERONET-OC. The data are divided by the direct-beam transmittance of the
other gases, as in hGRS; the transmittances are those of :py:class:`hgrs.gas_coupling.GasTransmittance`
(coefficient table ``HGRS_PPLD_TABLE`` if set) convolved with the EnMAP bands (Gaussian, FWHM of the product).

Models (W the water vapour column, wl in micron):

* ``decoupled``: :math:`T_{dir}(W)(a\lambda + b)` on 800-1300 nm (hGRS 1.2);
* ``path``: :math:`T_{atm}(W)\, c\lambda^{-n}` (path reflectance only) on ``RANGE``;
* ``path_surf``: :math:`T_{atm}(W)\, c\lambda^{-n} + T_{w}(W)(a\lambda + b)`, :math:`a\lambda + b \geq 0`;
* ``mix``: :math:`[f T_{atm}(W) + (1 - f) T_{dir}(W)]\, c\lambda^{-n}`, :math:`f \in [0, 1]` free.

Usage: python wv_experiments.py [wl_min wl_max]
'''
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from hgrs.gas_coupling import GasTransmittance  # noqa: E402
from common import OUT_DIR  # noqa: E402

warnings.filterwarnings('ignore')
GAS_LUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hgrs', 'data', 'lut',
                       'lut_abs_opt_thickness_normalized.nc')
W_GRID = np.array([0, 1, 2, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 60.])
RANGE = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (860., 1000.)


def bands(wl_f, values, centers, fwhms):
    out = []
    for c, f in zip(centers, fwhms):
        s = f / 2.3548
        w = np.exp(-0.5 * ((wl_f - c) / s) ** 2)
        w[np.abs(wl_f - c) > 3 * f] = 0
        out.append((values * w).sum(-1) / w.sum())
    return np.stack(out, -1)


def main():
    from closure_stats import load
    ds = load()
    g = xr.open_dataset(GAS_LUT)
    gt = GasTransmittance(g, 0.35, 'path_length', 'lbl', ppld_file=os.environ.get('HGRS_PPLD_TABLE'))
    rows = []
    for i in range(ds.sizes['matchup']):
        d = ds.isel(matchup=i)
        wl, fw = d.wl_band.values, d.fwhm.values
        r = d.Rtoa_enmap.values
        cv = float(d.Rtoa_enmap_std[np.argmin(abs(wl - 865))] / r[np.argmin(abs(wl - 865))])
        if not np.isfinite(d.tcwv) or cv > 0.15:
            continue
        M = 1 / np.cos(np.radians(float(d.sza))) + 1 / np.cos(np.radians(float(d.vza)))
        aot = float(d.aot550)
        pressure = float(d.pressure) if np.isfinite(d.pressure) else 1013.
        args = (M, pressure, 0.0067, 2e-6, 0.0105)
        t_other = bands(gt.wl, gt.other_gases('dir', *args), wl, fw)
        tab = {k: bands(gt.wl, np.stack([gt.total(k, *args, w, aot) for w in W_GRID]), wl, fw) / t_other
               for k in ('dir', 'atm', 'w')}
        # signal above the water vapour (cirrus, offset): median of the saturated bands of 1360-1390 and
        # 1870-1900 nm, subtracted (spectrally flat) when OFFSET is set
        sat = [int(np.argmin(abs(wl - w))) for w in (1360, 1370, 1380, 1390, 1870, 1880, 1890, 1900)]
        off = float(np.nanmedian(r[sat])) if os.environ.get('OFFSET') else 0.
        y_all = (r - off) / t_other
        T = lambda k, W, idx: np.array([np.interp(W, W_GRID, tab[k][:, j]) for j in idx])
        out = dict(matchup=str(d.matchup.values), site=str(d.site.values), W_aer=float(d.tcwv), aot=aot, M=M, off=off)
        # decoupled 800-1300 (hGRS 1.2)
        idx = np.flatnonzero((wl >= 800) & (wl <= 1300) & np.isfinite(y_all))
        x, y = wl[idx] / 1000, y_all[idx]
        f = least_squares(lambda p: T('dir', p[0], idx) * (p[1] * x + p[2]) - y, [20, -0.04, 0.1],
                          bounds=([0, -10, 0], [60, 1, 1]))
        out['decoupled'] = f.x[0]
        idx = np.flatnonzero((wl >= RANGE[0]) & (wl <= RANGE[1]) & np.isfinite(y_all))
        x, y = wl[idx] / 1000, y_all[idx]
        p0 = [20, float(y.max()), 1.]
        f = least_squares(lambda p: T('atm', p[0], idx) * p[1] * x ** -p[2] - y, p0, bounds=([0, 0, -4], [60, 1, 8]))
        out['path'] = f.x[0]
        f = least_squares(lambda p: T('atm', p[0], idx) * p[1] * x ** -p[2] + T('w', p[0], idx)
                          * np.maximum(p[3] * x + p[4], 0) - y, [*p0, 0., 0.], bounds=([0, 0, -4, -1, -1], [60, 1, 8, 1, 1]))
        out['path_surf'] = f.x[0]
        f = least_squares(lambda p: (p[3] * T('atm', p[0], idx) + (1 - p[3]) * T('dir', p[0], idx))
                          * p[1] * x ** -p[2] - y, [*p0, 0.8], bounds=([0, 0, -4, 0], [60, 1, 8, 1]))
        out['mix'], out['mix_f'] = f.x[0], f.x[3]
        f = least_squares(lambda p: T('dir', p[0], idx) * p[1] * x ** -p[2] - y, p0, bounds=([0, 0, -4], [60, 1, 8]))
        out['dir_range'] = f.x[0]
        rows.append(out)
    t = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(t.drop(columns='matchup').round(2).to_string())
    print(f'\nratio W_retrieved / W_AERONET (median, [p25, p75]), fit range {RANGE} nm, N = {len(t)}')
    for k in ('decoupled', 'dir_range', 'path', 'path_surf', 'mix'):
        q = (t[k] / t.W_aer).quantile([.25, .5, .75]).values
        print(f'  {k:10s} {q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]   rmsd {np.sqrt(np.mean((t[k] - t.W_aer) ** 2)):.1f} kg m-2')


if __name__ == '__main__':
    main()
