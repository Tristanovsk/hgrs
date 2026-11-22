'''
O2-A band of EnMAP over bright bare land: band ratio of the bare pixels against the line-by-line O2 transmittance.

Over a bright surface the TOA signal is mostly the surface term, with the direct-beam gaseous transmittance (the
coupling with the scattering raises the band transmittance by 0.4-0.9 % only at an albedo of 0.2-0.3, SOS-ABS). The
ratio of each band of the O2-A band (756, 763.7, 771 nm) to the linear continuum of the 749 and 778 nm bands is
divided by the same ratio of the line-by-line two-way direct transmittance of O2 (``hgrs.gas_coupling.O2LineByLine``,
1013.25 hPa, median air mass of the scene) convolved with the Gaussian bands of the metadata.

Pixels (every second one): bare, 0.02 < NDVI < 0.12 and :math:`R_{865}/R_{778}` > 0.95 (no water, no vegetation),
flat continuum :math:`|R_{778}/R_{749} - 1|` < 0.12; ``bright`` subset: :math:`R_{778}` > 0.15. Appends to
``closure/land_o2a.csv`` (one row per scene, bright pixels) and ``closure/land_o2a_bins.csv`` (one row per scene and
bin of :math:`R_{778}`, with the coupling correction ``fcorr`` of SOS-ABS interpolated on the estimated albedo
:math:`(R_{778} - 0.015)/0.85`).

Usage: python o2a_land.py <EnMAP L1C directory> ...    (~2 GB per scene)
'''
import os
import sys
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_DIR

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
CLOSURE = OUT_DIR / 'closure'
BANDS = dict(r665=665., l=749.06, a=756.35, b=763.70, c=771.11, r=778.57, n865=865.)
BINS = [0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20, 0.25, 0.35, 0.6]
BRIGHT = 0.15
# SOS-ABS v4 (radcalnet_oc), EnMAP 763.7 nm, total TOA signal over a Lambertian surface: band transmittance / that
# of the direct beam, AOT 0.05-0.2, SZA 20-45 (mean), against the albedo
FCORR_ALBEDO = [0.0, 0.03, 0.1, 0.2, 0.3, 0.4]
FCORR = [1.16, 1.06, 1.02, 1.009, 1.005, 1.004]
WL = np.arange(735., 800., 0.005)


def gaussian(v, c, f):
    s = f / 2.3548
    w = np.exp(-0.5 * ((WL - c) / s) ** 2)
    w[np.abs(WL - c) > 3 * f] = 0
    return (v * w).sum() / w.sum()


def depth(r, wl, k):
    return r[k] / (r['l'] + (r['r'] - r['l']) * (wl[k] - wl['l']) / (wl['r'] - wl['l']))


def append(df, name):
    f = CLOSURE / name
    df.to_csv(f, mode='a', header=not f.exists(), index=False, float_format='%.5g')


def scene(path, lbl):
    from hgrs.drivers.enmap import EnmapDriver
    ds = EnmapDriver().read(path)
    idx = {k: int(np.argmin(np.abs(ds.wl.values - v))) for k, v in BANDS.items()}
    wl = {k: float(ds.wl[i]) for k, i in idx.items()}
    fwhm = {k: float(ds.fwhm[i]) for k, i in idx.items()}
    sub = ds.isel(x=slice(None, None, 2), y=slice(None, None, 2))
    R = {k: sub.Rtoa.isel(wl=i).values for k, i in idx.items()}
    m = (1 / np.cos(np.radians(float(np.nanmedian(sub.sza.values))))
         + 1 / np.cos(np.radians(float(np.nanmedian(sub.vza.values)))))
    t = lbl.transmittance(m * 1013.25 / lbl.p_ref)
    tb = {k: gaussian(t, wl[k], fwhm[k]) for k in 'labcr'}
    ratio = {k: depth(R, wl, k) / depth(tb, wl, k) for k in 'abc'}
    ndvi = (R['n865'] - R['r665']) / (R['n865'] + R['r665'])
    bare = ((ndvi > 0.02) & (ndvi < 0.12) & (R['n865'] / R['r'] > 0.95) & (np.abs(R['r'] / R['l'] - 1) < 0.12)
            & np.isfinite(R['b']))
    name = os.path.basename(os.path.normpath(path))
    info = dict(scene=name, version=name.split('_V')[-1][:6], airmass=m)
    sel = bare & (R['r'] > BRIGHT)
    if sel.sum() >= 50:
        append(pd.DataFrame([dict(**info, n=int(sel.sum()), R778=float(np.median(R['r'][sel])),
                                  **{f'ratio_{k}': float(np.median(ratio[k][sel])) for k in 'abc'})]), 'land_o2a.csv')
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = bare & (R['r'] >= lo) & (R['r'] < hi)
        if sel.sum() >= 30:
            r778 = float(np.median(R['r'][sel]))
            rows.append(dict(**info, bin_lo=lo, n=int(sel.sum()), R778=r778,
                             fcorr=float(np.interp((r778 - 0.015) / 0.85, FCORR_ALBEDO, FCORR)),
                             **{f'ratio_{k}': float(np.median(ratio[k][sel])) for k in 'abc'}))
    if rows:
        append(pd.DataFrame(rows), 'land_o2a_bins.csv')
    print(name, int((bare & (R['r'] > BRIGHT)).sum()), 'bright pixels,', len(rows), 'bins', flush=True)


if __name__ == '__main__':
    from hgrs.gas_coupling import O2LineByLine
    lbl = O2LineByLine(WL)
    for p in sys.argv[1:]:
        try:
            scene(p, lbl)
        except Exception as e:
            print('fail', p, repr(e)[:200], flush=True)
