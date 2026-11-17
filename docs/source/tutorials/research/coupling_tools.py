'''
Helpers of the research notebooks on the coupling of gaseous absorption and scattering
(``coupling_*.ipynb``): SOS-ABS simulations, band averages, transmittance models of hGRS and of
radcalnet_oc, and a common figure style.

Data (local, not part of the repository; set the environment variables to use other folders):

* ``HGRS_SOS_ABS_DIR``: SOS-ABS coupled simulations, set v2 of
  ``radcalnet_oc/scripts/sos_abs_coupling/make_luts.py`` (``abs.nc``, ``noabs.nc``, ``tgdir_abs.nc``,
  ``tgdir_noabs.nc``), default ``/data/sos_abs_coupling/v2``;
* ``enmap_lucinda_bands.nc`` (next to this file): centres and FWHM of the EnMAP bands of the test scene.

radcalnet_oc (``pip install radcalnet_oc`` or an editable install) provides the photon path-length
coefficients (``gas_coupling_ppld.csv``) and the line-by-line O2 table.
'''
import os
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

import hgrs
import radcalnet_oc.kernel as rk

HERE = Path(__file__).resolve().parent
SOS_DIR = Path(os.environ.get('HGRS_SOS_ABS_DIR', '/data/sos_abs_coupling/v2'))
HGRS_GAS_FILE = Path(hgrs.__file__).parent / 'data' / 'lut' / 'lut_abs_opt_thickness_normalized.nc'
BANDS_FILE = HERE / 'enmap_lucinda_bands.nc'

# absorption windows sampled by the SOS-ABS v2 set (nm)
WINDOWS = {'O2-B': (684, 698), 'H2O-720': (712, 736), 'O2-A': (757, 772), 'H2O-820': (808, 836),
           'H2O-940': (890, 990)}

# ---------------------------------------------------------------------------------------------
# figure style: reference categorical palette (validated: lightness, chroma, CVD separation),
# fixed order; line styles and markers as secondary encoding; recessive grid
# ---------------------------------------------------------------------------------------------
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
STYLES = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'


def set_style():
    mpl.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 150, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.8,
        'lines.markersize': 6, 'legend.frameon': False, 'legend.fontsize': 9, 'text.color': INK,
        'axes.prop_cycle': mpl.cycler(color=COLORS),
    })


def series(i):
    '''Colour, line style and marker of the i-th series (fixed order).'''
    return dict(color=COLORS[i], linestyle=STYLES[i], marker=MARKERS[i])


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------
def air_mass(sza, vza=None):
    '''Geometric air mass 1/cos(sza) (+ 1/cos(vza) for the two-way path).'''
    m = 1 / np.cos(np.radians(sza))
    return m if vza is None else m + 1 / np.cos(np.radians(vza))


# ---------------------------------------------------------------------------------------------
# SOS-ABS simulations
# ---------------------------------------------------------------------------------------------
def load_sos(directory=SOS_DIR):
    '''SOS-ABS v2 set: TOA fields with / without absorption and direct-beam fluxes (wl in nm).'''
    out = []
    for name in ('abs', 'noabs', 'tgdir_abs', 'tgdir_noabs'):
        d = xr.open_dataset(directory / f'{name}.nc')
        d['wl'] = np.round(d.wl * 1000, 3)
        d.wl.attrs['units'] = 'nm'
        out.append(d)
    return tuple(out)


def toa(ds, sza, vza, azi, **sel):
    '''TOA reflectance pi L / (E0 cos sza) of a SOS-ABS field at a geometry (nearest values of `sel`).

    ``I_up`` of SOS-ABS is normalised to E0 on a surface normal to the beam: divided by cos(sza)
    to give the reflectance of hGRS (Rtoa = pi L / (E0 cos sza)).'''
    i = ds.I_up.sel(**sel, method='nearest') if sel else ds.I_up
    i = i.interp(sza=sza, vza=vza).sel(azi=azi, method='nearest')
    i = i.drop_vars(['azi', 'gauss_weight', 'user_angle', 'sza', 'vza'], errors='ignore')
    return i / np.cos(np.radians(sza))


def tg_direct(tgdir_abs, tgdir_noabs, m, h2o):
    '''SOS-ABS direct-beam gaseous transmittance for the air mass m (ratio of the direct fluxes at the
    surface, interpolated in m = 1/cos(sza)), water vapour column h2o (g cm-2, nearest).'''
    t = tgdir_abs.flux_dir_down.sel(h2o=h2o, method='nearest') / tgdir_noabs.flux_dir_down
    t = t.assign_coords(m=('sza', air_mass(t.sza.values))).swap_dims(sza='m').drop_vars(['sza', 'h2o'],
                                                                                         errors='ignore')
    return t.interp(m=m)


class DirectBeam:
    '''SOS-ABS direct-beam transmittance of the bands as a function of the slant water vapour path
    x = m U (g cm-2): along the direct beam the transmittance depends on the product only. The three
    columns of the simulations (0.5, 2, 4 g cm-2) and the 14 solar angles are merged (x from 0.5 to 40),
    with T = 1 at x = 0.

    :param bands: :py:class:`Bands`; :param centres: band centres, inside the SOS-ABS windows
    '''

    def __init__(self, tgdir_abs, tgdir_noabs, bands, centres):
        t = tgdir_abs.flux_dir_down / tgdir_noabs.flux_dir_down
        m = air_mass(t.sza.values)
        xs, ts = [np.zeros(1)], [np.ones((1, len(centres)))]
        for u in t.h2o.values:
            tb = bands.convolve_windows(t.sel(h2o=u)).sel(wl=list(centres), method='nearest')
            xs.append(m * u)
            ts.append(tb.transpose('sza', 'wl').values)
        x, tx = np.concatenate(xs), np.concatenate(ts)
        o = np.argsort(x)
        self.x, self.t, self.centres = x[o], tx[o], np.asarray(centres)

    def __call__(self, x):
        '''band transmittances for the slant path x = m U (g cm-2)'''
        return np.array([np.interp(x, self.x, self.t[:, i]) for i in range(len(self.centres))])


# ---------------------------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------------------------
class Bands:
    '''Gaussian spectral responses (centres and FWHM in nm), as hGRS for EnMAP
    (``SensorDescription.convolve``: same band averages, checked to 1e-3).'''

    def __init__(self, centres, fwhm):
        self.centres = np.asarray(centres, float)
        self.fwhm = np.asarray(fwhm, float)

    @classmethod
    def enmap_lucinda(cls, wl_min=None, wl_max=None):
        b = xr.open_dataset(BANDS_FILE).sel(wl=slice(wl_min, wl_max))
        return cls(b.wl.values, b.fwhm.values)

    def weights(self, wl, centres=None):
        centres = self.centres if centres is None else np.atleast_1d(centres)
        fwhm = np.interp(centres, self.centres, self.fwhm)
        w = np.exp(-0.5 * ((np.asarray(wl)[None, :] - centres[:, None]) / (fwhm[:, None] / 2.3548)) ** 2)
        return w / w.sum(1, keepdims=True)

    def average(self, values, wl, centres=None):
        '''Band averages of `values` (last axis on `wl`, any sampling as long as it is fine enough).'''
        return np.asarray(values) @ self.weights(wl, centres).T

    def inside(self, lo, hi, margin=0.85):
        '''Bands whose response (+-margin FWHM, about 2 sigma) is inside [lo, hi].'''
        half = margin * self.fwhm
        return self.centres[(self.centres - half >= lo) & (self.centres + half <= hi)]

    def in_windows(self, windows=WINDOWS, margin=0.85):
        '''{window: band centres} of the bands inside each SOS-ABS window.'''
        return {k: c for k, (lo, hi) in windows.items() if len(c := self.inside(lo, hi, margin))}

    def convolve_windows(self, da, windows=WINDOWS, margin=0.85):
        '''Band averages of a DataArray on the SOS-ABS wavelengths, for the bands inside the windows
        (``wl`` dimension of the result: band centres).'''
        out = []
        for name, centres in self.in_windows(windows, margin).items():
            lo, hi = windows[name]
            s = da.sel(wl=slice(lo, hi))
            w = xr.DataArray(self.weights(s.wl.values, centres), dims=('band', 'wl'),
                             coords={'band': centres, 'wl': s.wl.values})
            out.append(xr.dot(s, w, dim='wl'))
        return xr.concat(out, 'band').rename(band='wl').sortby('wl')


def on_fine(da, fine, windows=('O2-B', 'O2-A'), fill=1.):
    '''Linear interpolation of a transmittance sampled in some SOS-ABS windows on a fine grid, `fill`
    (1: no absorption) outside these windows. For band averages of bands wider than the windows.'''
    out = np.full(fine.shape, fill)
    for name in windows:
        lo, hi = WINDOWS[name]
        s = da.sel(wl=slice(lo, hi))
        ok = (fine >= lo) & (fine <= hi)
        out[ok] = np.interp(fine[ok], s.wl.values, s.values)
    return out


# ---------------------------------------------------------------------------------------------
# hGRS transmittance models (REPTRAN fine table of hGRS, native grid)
# ---------------------------------------------------------------------------------------------
def gas_lut(wl_min=None, wl_max=None):
    '''Normalized absorption optical thickness of hGRS (REPTRAN fine, per kg m-2 for h2o).'''
    return xr.open_dataset(HGRS_GAS_FILE).sel(wl=slice(wl_min, wl_max))


def hgrs_twv(m, tcwv, gas):
    '''hGRS water vapour transmittance exp(-m W kappa_h2o) on the native grid of `gas`, W in kg m-2.'''
    return np.exp(-m * tcwv * gas.h2o.values)


def hgrs_tg_o2(m, gas, kappa=0.35, pressure=1013.):
    '''hGRS transmittance of the O2, O4 and CO2 background (decoupled, coef_abs_scat kappa):
    exp(-m kappa (P/1000) (tau_O2 + tau_O4 + tau_CO2)), native grid of `gas`.'''
    tau = (gas.o2 + gas.o4 + gas.co2).values * pressure / 1000
    return np.exp(-m * kappa * tau)


# ---------------------------------------------------------------------------------------------
# radcalnet_oc photon path-length model (coefficients fitted on SOS-ABS)
# ---------------------------------------------------------------------------------------------
class PathLength:
    '''Photon path-length model of radcalnet_oc (``gas_coupling_ppld.csv``): per window and term,
    T = (1 + s2 eta m tau)^(-1/s2), tau the vertical absorption optical thickness of the channel.
    Terms: ``'atm'`` (path reflectance), ``'w'`` (water-leaving), ``'Ed'`` (irradiance).'''

    def __init__(self):
        self.c = rk.GasCoupling(rk.GAS_COUPLING_PPLD_FILE)
        self.windows = self.c.windows

    def window_of(self, wl, gas='H2O'):
        for name, (lo, hi) in self.windows.items():
            if name.startswith(gas) and lo <= wl < hi:
                return name
        # outside the fitted windows: nearest window of the same gas
        names = [k for k in self.windows if k.startswith(gas)]
        return min(names, key=lambda k: min(abs(wl - self.windows[k][0]), abs(wl - self.windows[k][1])))

    def window_masks(self, wl, gas='H2O'):
        '''[(window, boolean mask of wl)] of the windows of a gas on a wavelength grid (cached per grid).'''
        key = (gas, wl.size, float(wl[0]), float(wl[-1]))
        cache = self.__dict__.setdefault('_masks', {})
        if key not in cache:
            names = np.array([self.window_of(w, gas) for w in wl])
            cache[key] = [(n, names == n) for n in np.unique(names)]
        return cache[key]

    def eta_s2(self, term, window, m, aot550, h2o_gcm2=None):
        key = (term, window)
        rows = self.__dict__.setdefault('_rows', {})
        if key not in rows:
            rows[key] = self.c.row(term, window)
        r = rows[key]
        mc = np.clip(m, r.m_min, r.m_max) - 2
        ac = np.clip(aot550, r.aot_min, r.aot_max) - 0.1
        lu = 0. if h2o_gcm2 is None else np.log(np.clip(h2o_gcm2, r.h2o_min, r.h2o_max) / self.c.H2O_REF)
        eta = r.c0 + r.c1 * mc + r.c2 * ac + r.c3 * lu
        s2 = np.maximum(r.s0 + r.s1 * mc + r.s2 * ac + r.s3 * lu, rk.S2_MIN)
        return eta, s2

    def transmittance(self, x, s2):
        '''x = eta m tau'''
        return rk.path_length_transmittance(x, s2)


def ppld_twv(m, tcwv, gas, aot550, term='atm', pl=None):
    '''Water vapour transmittance of a term with the path-length model, on the native grid of `gas`
    (coefficients of the window of each wavelength), W in kg m-2.'''
    pl = pl or PathLength()
    wl = gas.wl.values
    out = np.empty(wl.size)
    for name, ok in pl.window_masks(wl, 'H2O'):
        eta, s2 = pl.eta_s2(term, name, m, aot550, tcwv / 10)
        out[ok] = pl.transmittance(eta * m * tcwv * gas.h2o.values[ok], s2)
    return out


def lbl_o2(fine, m, pressure=1013.):
    '''Line-by-line O2 direct transmittance (radcalnet_oc), cell averages on the grid `fine`.'''
    lbl = rk.O2LineByLine(fine)
    return lbl.transmittance(np.array([m * pressure / lbl.p_ref]))[0]


def ppld_lbl_o2(fine, m, aot550, term='atm', pressure=1013., pl=None):
    '''Line-by-line O2 transmittance of a term with the path-length model (radcalnet_oc O2 windows),
    1 outside the O2-B and O2-A windows.'''
    pl = pl or PathLength()
    out = np.ones(fine.shape)
    for name in ('O2-B', 'O2-A-R', 'O2-A-P'):
        lo, hi = pl.windows[name]
        ok = (fine >= lo) & (fine < hi)
        eta, s2 = pl.eta_s2(term, name, m, aot550)
        sub = rk.O2LineByLine(fine[ok])
        out[ok] = sub.transmittance_path_length(np.array([m * pressure / sub.p_ref]), np.array([eta]),
                                                np.array([s2]), np.zeros((1, ok.sum())))[0]
    return out
