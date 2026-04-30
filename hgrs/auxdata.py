import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


import importlib_resources as resources

# ******************************************************************************************************
dir, filename = os.path.split(__file__)
base_folder = resources.files("data").joinpath("aux")
thuillier_file = base_folder.joinpath("ref_atlas_thuillier3.nc")
gueymard_file = base_folder.joinpath("NewGuey2003.dat")
kurucz_file = base_folder.joinpath("kurucz_0.1nm.dat")
tsis_file = base_folder.joinpath(
    "hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc",
)

sunglint_eps_file = base_folder.joinpath(
    "mean_rglint_small_angles_vza_le_12_sza_le_60.txt"
)
rayleigh_file = base_folder.joinpath("rayleigh_bodhaine.txt")


class AuxData:
    """
    Describe the solar irradiance, rayleigh optical depth, and glint characteristics

    Args:
        - pressure_rot_ref: fixed reference pressure
        - rot (xr.DataArray): Rayleigh Optical Thickness (nominal pressure, temperature, c02) -> dim=
        - solar_irr (SolarIrradiance): solar irradiance
        - sunglint_eps (xr.DataArray)
    """

    def __init__(self, sensor_mod):
        # load data from raw files
        self.sunglint_eps = pd.read_csv(
            sunglint_eps_file, sep=r"\s+", index_col=0
        ).to_xarray()["mean"]
        self.rayleigh()
        self.pressure_rot_ref = 1013.25

    def rayleigh(self):
        """
        Rayleigh Optical Thickness for
        P=1013.25mb,
        T=288.15K,
        CO2=360ppm
        from
        Bodhaine, B.A., Wood, N.B, Dutton, E.G., Slusser, J.R. (1999). On Rayleigh
        Optical Depth Calculations, J. Atmos. Ocean Tech., 16, 1854-1861.
        """
        data = pd.read_csv(rayleigh_file, skiprows=16, sep=" ", header=None)
        data.columns = ("wl", "rot", "dpol")
        self.rot = data.set_index("wl").to_xarray().rot


class SolarIrradiance:
    def __init__(self, wl=None):
        # load data from raw files
        self.wl_min = 300
        self.wl_max = 2600

        self.gueymard = self.read_gueymard()
        self.kurucz = self.read_kurucz()
        self.thuillier = self.read_thuillier()
        self.tsis = self.read_tsis()

    def read_tsis(self):
        """
        Open TSIS data and convert them into xarray in mW/m2/nm
        :return:
        """
        tsis = xr.open_dataset(tsis_file)
        tsis = tsis.set_index(wavelength="Vacuum Wavelength").rename(
            {"wavelength": "wl"}
        )  # set_coords('Vacuum Wavelength')
        # convert
        tsis["SSI"] = tsis.SSI * 1000  # .plot(lw=0.5)
        tsis.SSI.attrs["units"] = "mW m-2 nm-1"
        tsis.SSI.attrs["long_name"] = (
            "Solar Spectral Irradiance Reference Spectrum (mW m-2 nm-1)"
        )
        tsis.SSI.attrs["reference"] = (
            "Coddington, O. M., Richard, E. C., Harber, D., et al. (2021)."
            + "The TSIS-1 Hybrid Solar Reference Spectrum. Geophysical Research Letters,"
            + "48(12), e2020GL091709. https://doi.org/10.1029/2020GL091709"
        )
        return tsis.SSI.sel(wl=slice(self.wl_min, self.wl_max))

    def read_thuillier(self):
        """
        Open Thuillier data and convert them into xarray in mW/m2/nm
        :return:
        """
        solar_irr = xr.open_dataset(thuillier_file).squeeze().data.drop("time") * 1e3
        solar_irr = solar_irr.rename({"wavelength": "wl"})
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[
            (solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)
        ]
        solar_irr.attrs["units"] = "mW/m2/nm"
        return solar_irr

    def read_gueymard(self):
        """
        Open Thuillier data and convert them into xarray in mW/m2/nm
        :return:
        """
        solar_irr = pd.read_csv(gueymard_file, sep=r"\s+", skiprows=30, header=None)
        solar_irr.columns = ["wl", "data"]
        solar_irr = solar_irr.set_index("wl").data.to_xarray()
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[
            (solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)
        ]
        solar_irr.attrs["units"] = "mW/m2/nm"
        solar_irr.attrs["reference"] = (
            "Gueymard, C. A., Solar Energy, Volume 76, Issue 4,2004, ISSN 0038-092X"
        )
        return solar_irr

    def read_kurucz(self):
        """
        Open Kurucz data and convert them into xarray in mW/m2/nm
        :return:
        """
        solar_irr = pd.read_csv(kurucz_file, sep=r"\s+", skiprows=11, header=None)
        solar_irr.columns = ["wl", "data"]
        solar_irr = solar_irr.set_index("wl").data.to_xarray()
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[
            (solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)
        ]
        solar_irr.attrs["units"] = "mW/m2/nm"
        solar_irr.attrs["reference"] = (
            "Kurucz, R.L., Synthetic infrared spectra, in Infrared Solar Physics, "
            + "IAU Symp. 154, edited by D.M. Rabin and J.T. Jefferies, Kluwer, Acad., "
            + "Norwell, MA, 1992."
        )
        return solar_irr

    # def interp(self, wl=[440, 550, 660, 770, 880]):
    #     """
    #     Interpolation on new wavelengths
    #     :param wl: wavelength in nm
    #     :return: update variables of the class
    #     """
    #     self.thuillier = self.thuillier.interp(wl=wl)
    #     self.gueymard = self.gueymard.interp(wl=wl)


def get_air_mass(vza, sza, round=True, digit_resol=3):
    air_mass = 1.0 / np.cos(np.radians(sza)) + 1.0 / np.cos(np.radians(vza))
    if round:
        return air_mass.round(digit_resol)
    return air_mass
