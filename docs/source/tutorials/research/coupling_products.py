'''
Metrics of L2A products processed in both gaseous transmittance modes, for the research notebooks
(``coupling_products_metrics.csv``): median TCWV, AOT and band depth of Rrs in the O2 and H2O bands.

Usage::

    python coupling_products.py SCENE DECOUPLED.nc COUPLED.nc [SCENE DECOUPLED.nc COUPLED.nc ...]

The products are made with ``hgrs_enmap`` (or another command), with a configuration file (``HGRS_CONFIG``) holding
only the LUT folder, so that no other user setting is mixed in: plus ``previous_behavior: true`` (decoupled, REPTRAN,
water vapor fit 800-1300 nm, i.e. hGRS 1.2.0) for the first one, the default settings of hGRS 2.0.0
(``path_length``, ``lbl_o2b``, 931-1060 nm) for the second one.
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT = Path(__file__).resolve().parent / 'coupling_products_metrics.csv'
# band, left and right neighbours (nm) of the band depth
DEPTHS = [(686.3, 679.7, 699.8), (693.0, 679.7, 699.8), (763.7, 749.1, 778.6), (720.5, 706.6, 741.8),
          (816.6, 778.6, 863.7), (941.2, 887.7, 993.1)]


def metrics(ds):
    '''median TCWV, AOT and band depth (Rrs(band) - linear continuum of the neighbours, sr-1)'''
    rrs = lambda w: ds.Rrs.sel(wl=w, method='nearest')
    out = dict(tcwv=float(ds.tcwv.median()), aot_ref=float(ds.aot_ref.median()))
    for b, l, r in DEPTHS:
        cont = rrs(l) + (rrs(r) - rrs(l)) * (b - l) / (r - l)
        out[f'depth {b}'] = float((rrs(b) - cont).median())
    for w in (443, 560, 665, 865):
        out[f'Rrs {w}'] = float(rrs(w).median())
    return out


def main(args):
    if len(args) % 3:
        sys.exit(__doc__)
    rows = []
    for scene, dec, cou in zip(args[::3], args[1::3], args[2::3]):
        for mode, path in (('decoupled', dec), ('coupled', cou)):
            with xr.open_dataset(path) as ds:
                rows.append(dict(scene=scene, mode=mode, **metrics(ds)))
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format='%.6g')
    print('written', OUT)


if __name__ == '__main__':
    main(sys.argv[1:])
