'''
TOA reflectance of the EnMAP VNIR and SWIR detectors at the same wavelength, in their overlap (975 and 990 nm), over
the darkest water pixels of a scene (10 % darkest at 1015 nm) and over land (median ratio SWIR / VNIR at 975 nm).

Water: R(1650) < 0.02 and R(560) > 3 R(1650); land: R(1650) > 0.1. Over dark water the SWIR detector reads 3e-3 to
1e-2 more than the VNIR one at the same wavelength (L1C V010502 and V010506), while the two agree to a few percent
over land (research notebook *Coupling of absorption and scattering 2/3*).

Usage: python detector_overlap.py l1c_path [l1c_path ...]
       python detector_overlap.py --matchups      (one scene per site and L1 processor version of matchups.csv)
'''
import re
import sys

import numpy as np
import pandas as pd

from common import OUT_DIR

KEEP = ((550, 570), (960, 1025), (1640, 1660))   # bands read (nm)


def overlap(path):
    from hgrs.drivers.enmap import EnmapDriver
    drv = EnmapDriver()
    R = drv.read(path, filter_bad_bands=False).Rtoa
    keep = np.array([any(lo < w < hi for lo, hi in KEEP) for w in R.wl.values])
    pos = np.cumsum(keep) - 1
    vi = pos[[i for i in drv.vnir_idx if keep[i]]]
    si = pos[[i for i in drv.swir_idx if keep[i]]]
    R = R.isel(wl=np.flatnonzero(keep)).load()
    wv, ws = R.wl.values[vi], R.wl.values[si]

    def band(idx, wls, w):
        return R.isel(wl=idx[np.argmin(abs(wls - w))])

    r1650 = R.sel(wl=1650, method='nearest')
    water = (r1650 < 0.02) & (R.sel(wl=560, method='nearest') > 3 * r1650)
    r1015 = R.sel(wl=1015, method='nearest').where(water)
    dark = water & (r1015 <= float(r1015.quantile(0.1)))
    land = r1650 > 0.1
    out = dict(scene=re.search(r'DT\d+_(\d{8})', path).group(1) + ' ' + path.rstrip('/').split('/')[-2],
               l1_version=re.search(r'_(V\d{6})_', path).group(1), n_dark=int(dark.sum()))
    for w in (975, 990):
        v, s = float(band(vi, wv, w).where(dark).median()), float(band(si, ws, w).where(dark).median())
        out.update({f'vnir_{w}': v, f'swir_{w}': s, f'swir-vnir_{w}': s - v})
    out['land_swir/vnir_975'] = (float(band(si, ws, 975).where(land).median() / band(vi, wv, 975).where(land).median())
                                 if int(land.sum()) > 100 else np.nan)
    return out


def main(args):
    if args == ['--matchups']:
        m = pd.read_csv(OUT_DIR / 'matchups.csv').drop_duplicates('product')
        m['version'] = m['product'].str.extract(r'_(V\d{6})_')[0]
        args = list(m.groupby(['site', 'version']).head(1).path)
    if not args:
        sys.exit(__doc__)
    rows = []
    for p in args:
        try:
            rows.append(overlap(p))
        except Exception as e:  # missing or unreadable product
            print(f'{p}: {e!r}', file=sys.stderr)
            continue
        print(pd.Series(rows[-1]).to_frame().T.to_string(header=len(rows) == 1, index=False,
                                                         float_format=lambda v: f'{v:.4f}'), flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main(sys.argv[1:])
