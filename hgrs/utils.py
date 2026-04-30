# copyright 2025,  Magellium, J.-P. Burochin

import numpy as np
import xarray as xr

import xesmf as xe
import logging


class Reproj:
    def __init__(self):
        pass

    @staticmethod
    def regridding(input_dataset, new_grid, parallel=True):
        """
        Take a PRISMA L1C product in sensor geometry (x,y) as input and
        return it in a georeferenced geometry (lon,lat).

        WARNING : Due to the use of the xESMF package, relying on Fortran,
        some user warnings like : "UserWarning: Input array is not F_CONTIGUOUS.
        Will affect performance." may be raised. It is not an issue in our case
        (see https://github.com/JiaweiZhuang/xESMF/issues/25).

        :param input_dataset: the product to regrid
        :param new_grid: grid

        :param d_input_crs: (int) code EPSG of the related geolocalisation frame

        :return output_dataset: the regularised product
        """
        logging.info("georeferencing native image")
        # setting lon and lat as coordinates

        # input_dataset = input_dataset.set_coords(["lon", "lat"])

        # use periodic=False if either or both the lat and lon dimensions are not regular
        regridder = xe.Regridder(
            input_dataset,
            new_grid,
            method="bilinear",
            periodic=False,
            unmapped_to_nan=True,
            parallel=parallel,
        )

        # regrid the data
        output_dataset = regridder(input_dataset)

        output_dataset = output_dataset.assign(fwhm=input_dataset.fwhm)

        return output_dataset


class Misc:
    """
    Miscelaneous utilities
    """

    @staticmethod
    def get_pressure(alt, psl):
        """Compute the pressure for a given altitude
        alt : altitude in meters (float or np.array)
        psl : pressure at sea level in hPa
        palt : pressure at the given altitude in hPa"""

        palt = psl * (1.0 - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    @staticmethod
    def transmittance_dir(aot, air_mass, rot=0):
        return np.exp(-(rot + aot) * air_mass)

    @staticmethod
    def air_mass(sza, vza):
        return 1 / np.cos(np.radians(vza)) + 1 / np.cos(np.radians(sza))

    @staticmethod
    def earth_sun_correction(dayofyear):
        """
        Earth-Sun distance correction factor for adjustment of mean solar irradiance

        :param dayofyear:
        :return: correction factor
        """
        theta = 2.0 * np.pi * dayofyear / 365
        d2 = (
            1.00011
            + 0.034221 * np.cos(theta)
            + 0.00128 * np.sin(theta)
            + 0.000719 * np.cos(2 * theta)
            + 0.000077 * np.sin(2 * theta)
        )
        return d2
