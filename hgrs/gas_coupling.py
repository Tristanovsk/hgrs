r'''
Coupling of the gaseous absorption with the scattering: gaseous transmittance of each term of the TOA signal.

The direct-beam transmittance :math:`T_g(\bar M) = \exp(-\bar M \tau_g)` is exact for the light crossing the
atmosphere along straight lines (sun glint, nearly the water-leaving signal). The photons of the **path
reflectance**, scattered by the molecules and aerosols above the absorbers, shortcut part of the absorbing
column and their paths are spread. Following ``radcalnet_oc`` (photon path-length model, fitted on SOS-ABS
coupled simulations), the transmittance of a term of geometric air mass :math:`m` is computed channel by
channel with a Gamma distribution of the absorber paths:

.. math::

   T = \left(1 + s^2\, \eta\, m\, \tau\right)^{-1/s^2}

where :math:`\tau` is the vertical absorption optical thickness of the channel, :math:`\eta` the ratio of the
mean absorber path to the geometric one and :math:`s^2` the relative variance of the paths, both linear in
:math:`m - 2`, :math:`\tau_{550} - 0.1` and :math:`\ln(U / 2\,\mathrm{g\,cm^{-2}})`, per absorption window
(O2-B, H2O 720, O2-A, H2O 820, 900 and 940 nm, 684-990 nm) and per term: ``'atm'`` (path reflectance) and
``'w'`` (water-leaving signal). Outside the windows, and for O3, NO2 and CH4 (above or away from the
scatterers), the direct-beam air mass is used.

In the O2-B and O2-A bands, the O2 absorption is computed line by line (:py:class:`O2LineByLine`): the
REPTRAN fine table, used as a spectrum at sub-nanometre resolution, makes these bands too deep.

Ported from ``radcalnet_oc`` (https://github.com/Tristanovsk/radcalnet_oc, Apache-2.0, commit d4f6c6f):
``kernel.path_length_transmittance``, ``kernel.GasCoupling`` (form ``ppld``) and ``kernel.O2LineByLine``;
data ``data/aux/gas_coupling_ppld.csv`` and ``data/lut/o2_lbl_tau.nc``. The analysis is in the research
notebooks of the documentation (*Coupling of absorption and scattering*).
'''

from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr

PPLD_FILE = files(__package__) / 'data' / 'aux' / 'gas_coupling_ppld.csv'
O2_LBL_FILE = files(__package__) / 'data' / 'lut' / 'o2_lbl_tau.nc'
S2_MIN = 1e-4  # minimum relative variance of the photon path-length distribution
H2O_REF = 2.  # g cm-2, reference water vapour column of the coefficients
GAS_COUPLINGS = ('path_length', 'decoupled')
O2_MODELS = ('lbl', 'reptran')
TERMS = ('atm', 'w')


def path_length_transmittance(x, s2):
    r'''
    Transmittance for a Gamma distribution of the absorber optical paths of mean :math:`x` and relative
    variance :math:`s^2`: :math:`(1 + s^2 x)^{-1/s^2}` (:math:`e^{-x}` when :math:`s^2 \to 0`).

    :param x: mean absorber optical path (:math:`\eta\, m\, \tau`)
    :param s2: relative variance of the distribution (clipped to ``S2_MIN``)
    '''
    s2 = np.maximum(s2, S2_MIN)
    return np.exp(-np.log1p(s2 * x) / s2)


class PathLengthCoupling:
    r'''
    Coefficients of the photon path-length model per window and term (``gas_coupling_ppld.csv``).

    :ivar table: :py:class:`pandas.DataFrame` (``window``, ``wl_min``, ``wl_max``, ``term``, ``c0``-``c3``,
        ``s0``-``s3``, validity ranges ``m_min``, ``m_max``, ``aot_min``, ``aot_max``, ``h2o_min``, ``h2o_max``)
    '''

    def __init__(self, file=PPLD_FILE):
        with open(file) as f:
            self.table = pd.read_csv(f, comment='#')
        self._rows = {(r.window, r.term): r for r in self.table.itertuples()}

    @property
    def windows(self):
        '''``{window: (wl_min, wl_max)}`` (nm)'''
        t = self.table.drop_duplicates('window')
        return {r.window: (float(r.wl_min), float(r.wl_max)) for r in t.itertuples()}

    def eta_s2(self, term, window, air_mass, aot550, h2o=None):
        r'''
        :math:`\eta` and :math:`s^2` of a term in a window (inputs clipped to the ranges of the fit).

        :param term: ``'atm'`` or ``'w'``
        :param air_mass: geometric air mass :math:`m` of the term (two-way)
        :param aot550: aerosol optical thickness at 550 nm
        :param h2o: water vapour column (g cm-2); None: 2 g cm-2
        '''
        r = self._rows[(window, term)]
        mc = np.clip(air_mass, r.m_min, r.m_max) - 2
        ac = np.clip(aot550, r.aot_min, r.aot_max) - 0.1
        lu = 0. if h2o is None else np.log(np.clip(h2o, r.h2o_min, r.h2o_max) / H2O_REF)
        eta = r.c0 + r.c1 * mc + r.c2 * ac + r.c3 * lu
        s2 = np.maximum(r.s0 + r.s1 * mc + r.s2 * ac + r.s3 * lu, S2_MIN)
        return eta, s2

    def masks(self, wl, gas):
        '''``[(window, mask)]`` of the windows of a gas (``'H2O'`` or ``'O2'``) on the wavelengths ``wl``'''
        wl = np.asarray(wl)
        return [(name, (wl >= lo) & (wl < hi)) for name, (lo, hi) in self.windows.items() if name.startswith(gas)]

    def coupled_mask(self, wl):
        '''wavelengths of ``wl`` inside a window'''
        wl = np.asarray(wl)
        out = np.zeros(wl.shape, dtype=bool)
        for lo, hi in self.windows.values():
            out |= (wl >= lo) & (wl < hi)
        return out


class O2LineByLine:
    r'''
    Line-by-line O2 transmittance in the O2-B (683-702 nm) and O2-A (755-775 nm) bands, averaged in the cells of
    a wavelength grid (``o2_lbl_tau.nc``: vertical optical thickness from HITRAN lines, Voigt profiles, AFGL
    midlatitude summer, :math:`P_{lbl}` = 1013 hPa, step 0.004 cm-1). The lines are narrower than the cells, so
    the transmittance of a path is averaged in each cell, :math:`\langle \exp(-f \tau_{O_2}) \rangle` with
    :math:`f = m P / P_{lbl}`.

    :param wl: wavelengths of the grid (nm), increasing
    '''

    def __init__(self, wl, file=O2_LBL_FILE):
        with xr.open_dataarray(file) as tau:
            tau = tau.load()
        self.p_ref = float(tau.attrs['surface_pressure'])
        wl = np.asarray(wl, dtype=float)
        edges = np.r_[wl[0] - (wl[1] - wl[0]) / 2, 0.5 * (wl[1:] + wl[:-1]), wl[-1] + (wl[-1] - wl[-2]) / 2]
        idx = np.digitize(tau.wl.values, edges) - 1
        ok = (idx >= 0) & (idx < len(wl))
        self.tau = tau.values.astype(float)[ok]
        idx = idx[ok]
        self.starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1] if len(idx) else np.zeros(0, dtype=int)
        self.cells = idx[self.starts]
        self.counts = np.diff(np.r_[self.starts, len(idx)])
        self.mask = np.zeros(len(wl), dtype=bool)
        self.mask[self.cells] = True
        self.nwl = len(wl)

    def transmittance(self, factor, eta=1., s2=0., tau_add=None):
        r'''
        Cell-averaged transmittance of the O2 lines (and of an absorption ``tau_add`` constant in each cell):
        direct beam :math:`\langle e^{-(\tau_a + f\tau)} \rangle` (``s2`` = 0) or path-length model
        :math:`\langle (1 + s^2 \eta (\tau_a + f \tau))^{-1/s^2} \rangle`, as ``radcalnet_oc``
        (``O2LineByLine.transmittance_path_length``).

        :param factor: :math:`f = m P / P_{lbl}` (scalar)
        :param eta, s2: parameters of the path-length model
        :param tau_add: slant optical thickness of the other absorbers in each cell (array of the grid);
            None: 0 (transmittance of the O2 lines only, 1 outside the bands)
        '''
        tau_add = np.zeros(self.nwl) if tau_add is None else np.asarray(tau_add, dtype=float)
        fun = (lambda x: np.exp(-x)) if s2 == 0 else (lambda x: path_length_transmittance(x, s2))
        out = fun(eta * tau_add)
        if len(self.cells):
            point_cell = np.repeat(self.cells, self.counts)
            t = fun(eta * (tau_add[point_cell] + factor * self.tau))
            out[self.cells] = np.add.reduceat(t, self.starts) / self.counts
        return out


class GasTransmittance:
    r'''
    Gaseous transmittance of the terms of the TOA signal on the fine grid of the gaseous LUT of hGRS
    (``lut_abs_opt_thickness_normalized.nc``), before the convolution with the band responses.

    * ``'dir'``: direct beam along the air mass :math:`\bar M` (sun glint, and the reference of hGRS);
    * ``'atm'``: path reflectance; ``'w'``: water-leaving signal (path-length model in the windows).

    :param gas_lut: gaseous LUT (normalized optical thicknesses, ``wl`` in nm)
    :param coef_abs_scat: factor :math:`\kappa` of the O2, O4 and CO2 optical thickness (hGRS
        ``gas_absorption_scattering_coefficient``), used where the coupling is not modelled
    :param coupling: ``'path_length'`` or ``'decoupled'`` (direct beam for all the terms)
    :param o2_model: ``'lbl'`` (line by line in the O2-B and O2-A bands) or ``'reptran'``
    '''

    def __init__(self, gas_lut, coef_abs_scat, coupling='path_length', o2_model='lbl'):
        if coupling not in GAS_COUPLINGS:
            raise ValueError(f'gas coupling must be one of {GAS_COUPLINGS}, not {coupling!r}')
        if o2_model not in O2_MODELS:
            raise ValueError(f'o2 model must be one of {O2_MODELS}, not {o2_model!r}')
        self.gas_lut = gas_lut
        self.wl = gas_lut.wl.values
        self.coef_abs_scat = coef_abs_scat
        self.coupling = coupling
        self.o2_model = o2_model
        self.ppld = PathLengthCoupling()
        self.coupled = self.ppld.coupled_mask(self.wl) if coupling == 'path_length' else np.zeros(self.wl.size, bool)
        self.o2_lbl = O2LineByLine(self.wl) if o2_model == 'lbl' else None
        self._h2o_masks = self.ppld.masks(self.wl, 'H2O')
        self._o2_masks = self.ppld.masks(self.wl, 'O2')
        # line-by-line O2 on the wavelengths of each O2 window (path-length model per window)
        self._o2_windows = [(name, ok, O2LineByLine(self.wl[ok])) for name, ok in self._o2_masks
                            if self.o2_lbl is not None and ok.sum() > 1]

    # optical thicknesses (vertical) ------------------------------------------------------------
    def tau_uncoupled(self, to3c, tno2c, tch4c):
        '''O3, NO2 and CH4 (kept on the direct-beam air mass)'''
        g = self.gas_lut
        return (g.o3 * to3c + g.no2 * tno2c + g.ch4 * tch4c).values

    def tau_background(self, pressure):
        r''':math:`(P/1000)[\tau_{CO} + \kappa(\tau_{CO_2} + \tau_{O_2} + \tau_{O_4})]`, the O2 lines of the O2-B
        and O2-A bands left out with ``o2_model='lbl'``; :math:`\kappa` = 1 in the coupled windows'''
        g = self.gas_lut
        kappa = np.where(self.coupled, 1., self.coef_abs_scat)
        o2 = g.o2.values
        if self.o2_lbl is not None:
            o2 = np.where(self.o2_lbl.mask, 0., o2)
        return (g.co.values + kappa * (g.co2.values + o2 + g.o4.values)) * pressure / 1000

    # transmittances -----------------------------------------------------------------------------
    def other_gases(self, term, air_mass, pressure, to3c, tno2c, tch4c, aot550=0.1, h2o=None):
        r'''
        Transmittance of the gases other than water vapour for a term (``'dir'``, ``'atm'`` or ``'w'``), fine grid.

        :param air_mass: two-way air mass :math:`\bar M`
        :param pressure: surface pressure (hPa)
        :param h2o: water vapour column (g cm-2), for the coefficients
        '''
        t = np.exp(-air_mass * self.tau_uncoupled(to3c, tno2c, tch4c))
        tau_bg = self.tau_background(pressure)
        if term == 'dir' or self.coupling == 'decoupled':
            t_bg = np.exp(-air_mass * tau_bg)
            if self.o2_lbl is not None:
                t_bg = t_bg * self.o2_lbl.transmittance(air_mass * pressure / self.o2_lbl.p_ref)
            return t * t_bg
        t_bg = np.exp(-air_mass * tau_bg)
        # background gases in the windows (O2 lines excepted, see below): path-length model
        for name, ok in self._h2o_masks + self._o2_masks:
            eta, s2 = self.ppld.eta_s2(term, name, air_mass, aot550, h2o)
            t_bg[ok] = path_length_transmittance(eta * air_mass * tau_bg[ok], s2)
        if self.o2_lbl is not None:
            # O2 lines and the other background gases together, point by point, in the O2 windows
            for name, ok, sub in self._o2_windows:
                eta, s2 = self.ppld.eta_s2(term, name, air_mass, aot550, h2o)
                t_bg[ok] = sub.transmittance(air_mass * pressure / sub.p_ref, eta, s2, tau_add=air_mass * tau_bg[ok])
            # O2 lines outside the windows (band wings), direct beam
            outside = self.o2_lbl.mask & ~self.coupled
            if outside.any():
                t_bg[outside] *= self.o2_lbl.transmittance(air_mass * pressure / self.o2_lbl.p_ref)[outside]
        return t * t_bg

    def water_vapor(self, term, air_mass, tcwv, aot550=0.1):
        r'''
        Water vapour transmittance of a term on the fine grid, :math:`W` = ``tcwv`` (kg m-2).
        '''
        tau = self.gas_lut.h2o.values * tcwv
        t = np.exp(-air_mass * tau)
        if term != 'dir' and self.coupling == 'path_length':
            for name, ok in self._h2o_masks:
                eta, s2 = self.ppld.eta_s2(term, name, air_mass, aot550, tcwv / 10.)
                t[ok] = path_length_transmittance(eta * air_mass * tau[ok], s2)
        return t

    def water_vapor_table(self, term, air_mass, tcwv_values, aot550=0.1):
        ''':py:meth:`water_vapor` for several columns: DataArray (tcwv, wl)'''
        tcwv_values = np.asarray(tcwv_values, dtype=float)
        t = np.stack([self.water_vapor(term, air_mass, w, aot550) for w in tcwv_values])
        return xr.DataArray(t, coords={'tcwv': tcwv_values, 'wl': self.wl}, dims=('tcwv', 'wl'), name='Twv')
