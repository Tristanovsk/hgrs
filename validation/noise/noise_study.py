'''
Instrument noise of PRISMA and EnMAP over water, for the noise model of the hGRS inversion.

Over the water pixels of a scene (criteria of hGRS: green-NIR and green-1600 nm indices, R(2200) < 0.11), in
TOA reflectance per band:

* ``sigma_random``: random noise, robust standard deviation (1.4826 MAD) of the difference of two pixels adjacent
  along track in the same detector column, divided by sqrt(2); the spatial variability of the water and of the
  glint over 30 m is included, so it is an upper bound of the noise;
* ``sigma_column``: fixed pattern of the detector columns (striping), robust standard deviation of the column
  medians after removal of their running median over 31 columns (PRISMA, native geometry only);
* ``sigma_crosstrack``: cross-track pattern at the scale of the mega-pixels of hGRS (20 columns), standard
  deviation of the column medians averaged over 20 columns after removal of a quadratic trend across the swath
  (radiometric non-uniformity, smile, and the cross-track gradients of glint and aerosol: an upper bound);
* ``signal``: median reflectance of the water pixels.

The noise of a mega-pixel mean of N = 400 pixels is sigma_random / 20 for the random part, but the column and
cross-track patterns do not average out along track: sigma_crosstrack sets the floor of the radiometric
uncertainty of the retrieval in each band, to compare with the tolerance of the non-negativity (``sigma_+``) and the
weight of the fit (``sigma_R``) of the inversion.

PRISMA L1 (``.he5``) is read in its native geometry (rows along track, columns = detector columns). EnMAP L1C is
orthorectified: its columns are not the detector columns and the resampling correlates neighbouring pixels, so only
``sigma_random`` (underestimated) is given.

Usage: python noise_study.py <L1 product> [...]      (writes noise_<product>.csv and .png in NOISE_OUT)
'''
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get('NOISE_OUT', '/data/satellite/noise_study'))


def robust_std(x, axis=None):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def convolved_f0(wl, fwhm, doy):
    from hgrs.auxdata import SolarIrradiance
    from hgrs.utils import Misc
    f0 = SolarIrradiance().get('tsis')
    w, v = f0.wl.values, f0.values * Misc.earth_sun_correction(doy)
    out = []
    for c, f in zip(wl, fwhm):
        s = f / 2.3548
        k = np.exp(-0.5 * ((w - c) / s) ** 2)
        out.append(np.sum(k * v) / np.sum(k))
    return np.array(out)


def read_prisma(path):
    '''TOA reflectance cube (row, col, band) in native geometry, wavelengths, FWHM'''
    import h5py
    with h5py.File(path, 'r') as f:
        a = f.attrs
        vn = f['/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube'][:, 5:, :] / a['ScaleFactor_Vnir']
        sw = f['/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube'][:, :-2, :] / a['ScaleFactor_Swir']
        wl = np.r_[a['List_Cw_Vnir'][5:], a['List_Cw_Swir'][:-2]]
        fw = np.r_[a['List_Fwhm_Vnir'][5:], a['List_Fwhm_Swir'][:-2]]
        sza = float(a['Sun_zenith_angle'])
        t0 = a.get('Product_StartTime', b'')
        date = (t0.decode() if isinstance(t0, bytes) else str(t0))[:10]
    L = np.concatenate([vn, sw], axis=1).astype('float32')          # (row, band, col)
    del vn, sw
    keep = wl > 0
    L, wl, fw = L[:, keep, :], wl[keep], fw[keep]
    order = np.argsort(wl)
    L, wl, fw = L[:, order, :], wl[order], fw[order]
    doy = pd.Timestamp(date).dayofyear if date else 180
    f0 = convolved_f0(wl, fw, doy)
    R = np.pi * L / (f0[None, :, None] * np.cos(np.radians(sza)))
    return np.moveaxis(R, 1, 2), wl, fw, True                        # (row, col, band), native geometry


def read_enmap(path):
    from hgrs.drivers.enmap import EnmapDriver
    ds = EnmapDriver().read(path)
    R = ds.Rtoa.transpose('y', 'x', 'wl').values.astype('float32')
    return R, ds.wl.values, ds.fwhm.values, False


def water_mask(R, wl):
    band = lambda lo, hi: np.nanmean(R[..., (wl >= lo) & (wl <= hi)], -1)
    g, nir, s16, s22 = band(540, 570), band(850, 882), band(1580, 1650), band(2150, 2250)
    return ((g - nir) / (g + nir) > 0.01) & ((g - s16) / (g + s16) > 0.1) & (s22 < 0.11)


def analyse(path):
    path = str(path)
    name = Path(path).name.replace('.he5', '')
    R, wl, fw, native = read_prisma(path) if path.endswith('.he5') else read_enmap(path)
    water = water_mask(R, wl)
    rows = []
    # adjacent along track, both water
    pair = water[1:, :] & water[:-1, :]
    ncol_min = 50
    colcount = water.sum(0)
    goodcol = colcount >= ncol_min
    for b in range(len(wl)):
        x = R[..., b]
        d = (x[1:, :] - x[:-1, :])[pair]
        sig_r = robust_std(d) / np.sqrt(2) if d.size > 100 else np.nan
        signal = np.nanmedian(x[water])
        sig_c = sig_l = np.nan
        if native and goodcol.sum() > 100:
            colmed = np.array([np.nanmedian(x[water[:, j], j]) if goodcol[j] else np.nan for j in range(x.shape[1])])
            s = pd.Series(colmed)
            trend = s.rolling(31, center=True, min_periods=10).median()
            sig_c = robust_std((s - trend).values)
            mp = s.rolling(20, center=True, min_periods=10).mean().values[::20]
            j = np.arange(len(mp))
            ok = np.isfinite(mp)
            if ok.sum() > 5:
                p = np.polyfit(j[ok], mp[ok], 2)
                sig_l = np.nanstd(mp[ok] - np.polyval(p, j[ok]))
        rows.append(dict(wl=wl[b], fwhm=fw[b], signal=signal, sigma_random=sig_r, sigma_column=sig_c,
                         sigma_crosstrack=sig_l, n_water=int(water.sum())))
    t = pd.DataFrame(rows)
    t['sigma_megapixel'] = np.sqrt((t.sigma_random / 20) ** 2 + np.nan_to_num(t.sigma_crosstrack) ** 2)
    t['snr'] = t.signal / t.sigma_random
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f'noise_{name}.csv', index=False, float_format='%.4g')
    plot(t, name, native)
    if native:
        np.save(OUT / f'colprofile_{name}.npy', np.array(
            [[np.nanmedian(R[water[:, j], j, b]) if goodcol[j] else np.nan for j in range(R.shape[1])]
             for b in [int(np.argmin(np.abs(wl - w))) for w in (490, 560, 750, 865, 1020, 1650, 2200)]]))
    print(name, f'{int(water.sum())} water pixels')
    print(t.iloc[[int(np.argmin(np.abs(t.wl - w))) for w in (443, 490, 560, 665, 750, 865, 1020, 1240, 1650, 2200)]]
          [['wl', 'signal', 'sigma_random', 'sigma_column', 'sigma_crosstrack', 'sigma_megapixel', 'snr']]
          .to_string(index=False, float_format=lambda v: f'{v:.3g}'))


def plot(t, name, native):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.semilogy(t.wl, t.signal, 'k-', lw=1, label='water signal (TOA)')
    ax.semilogy(t.wl, t.sigma_random, '.', ms=3, label='random (pixel)')
    if native:
        ax.semilogy(t.wl, t.sigma_column, '.', ms=3, label='column pattern')
        ax.semilogy(t.wl, t.sigma_crosstrack, '.', ms=3, label='cross-track (20 columns)')
    ax.semilogy(t.wl, t.sigma_megapixel, '-', lw=1, label='mega-pixel total')
    ax.axhline(2e-3, color='r', ls='--', lw=0.8, label=r'$\sigma_+$ = 2e-3')
    ax.axhline(1e-3, color='b', ls=':', lw=0.8, label=r'$\sigma_R$ = 1e-3')
    ax.set(xlabel='Wavelength (nm)', ylabel='TOA reflectance', ylim=(1e-5, 0.3), title=name[:60])
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f'noise_{name}.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyse(p)
