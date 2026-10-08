# copyright 2025,  Magellium, J.-P. Burochin

import numpy as np
import xarray as xr

import xesmf as xe
import logging

class Reproj():
    def __init__(self):
        pass

    @staticmethod
    def regridding(input_dataset,
                   output_grid_size=(1200, 1200),
                   d_input_crs=4326,
                   parallel=True):
        """
        Take a PRISMA L1C product in sensor geometry (x,y) as input and
        return it in a georeferenced geometry (lon,lat), by bilinear interpolation with xESMF
        (https://github.com/pangeo-data/xESMF) onto a regular longitude-latitude grid covering the image.

        WARNING : Due to the use of the xESMF package, relying on Fortran,
        some user warnings like : "UserWarning: Input array is not F_CONTIGUOUS.
        Will affect performance." may be raised. It is not an issue in our case
        (see https://github.com/JiaweiZhuang/xESMF/issues/25).

        :param input_dataset: the product to regrid
        :param output_grid_size: (tuple) output grid size in (lon, lat) format
        :param d_input_crs: (int) code EPSG of the related geolocalisation frame

        :return output_dataset: the regularised product

        Examples
        --------
        .. code-block:: python

           from hgrs import Reproj

           l1c_geo = Reproj.regridding(l1c, output_grid_size=(1200, 1200))   # done by Driver.read_prisma
        """

        logging.info('georeferencing native image')
        # setting lon and lat as coordinates
        attrs = input_dataset.attrs
        #input_dataset = input_dataset.set_coords(["lon", "lat"])

        # make the grid that the data will be regridded to
        grid_lons = np.linspace(input_dataset.lon.min().values, input_dataset.lon.max().values, output_grid_size[0])
        grid_lats = np.linspace(input_dataset.lat.min().values, input_dataset.lat.max().values, output_grid_size[1])
        new_grid = xr.Dataset({'lat': (['lat'], grid_lats), 'lon': (['lon'], grid_lons)})
        new_grid = new_grid.chunk({"lat": 50, "lon": 50})

        # use periodic=False if either or both the lat and lon dimensions are not regular
        regridder = xe.Regridder(input_dataset, new_grid,
                                 method='bilinear',
                                 periodic=False,
                                 unmapped_to_nan=True,
                                 parallel=parallel)

        # regrid the data
        output_dataset = regridder(input_dataset)

        # put the wavelength dependant data lost in the process, back in the dataset
        output_dataset = output_dataset.assign(fwhm=input_dataset.fwhm, F0=input_dataset.F0)

        # put "x","y" naming:
        output_dataset = output_dataset.rename({"lon": "x", "lat": "y"})

        # adding the CRS
        output_dataset.rio.write_crs(d_input_crs, inplace=True)
        output_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        output_dataset.rio.write_coordinate_system(inplace=True)
        output_dataset.attrs.update(attrs)
        return output_dataset



class Misc:
    '''
    Miscelaneous utilities
    '''

    @staticmethod
    def get_pressure(alt, psl):
        r'''Compute the pressure for a given altitude (barometric formula)

        .. math::

           P(z) = P_{sl} \left(1 - \frac{0.0065\, z}{288.15}\right)^{5.255}

        :param alt: altitude :math:`z` in meters (float or np.array)
        :param psl: pressure at sea level :math:`P_{sl}` in hPa
        :return: pressure at the given altitude in hPa

        The constants are those of the troposphere of the U.S. Standard Atmosphere (1976): lapse rate
        0.0065 K m\ :sup:`-1` and sea-level temperature 288.15 K.

        Examples
        --------
        >>> from hgrs.utils import Misc
        >>> round(float(Misc.get_pressure(500, 1013.25)), 2)
        954.62
        '''

        palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    @staticmethod
    def transmittance_dir(aot, air_mass, rot=0):
        r'''Direct transmittance :math:`T_{dir} = \exp\left[-(\tau_R + \tau_a) M\right]`

        :param aot: aerosol optical thickness :math:`\tau_a`
        :param air_mass: air mass :math:`M`
        :param rot: Rayleigh optical thickness :math:`\tau_R`

        Examples
        --------
        >>> from hgrs.utils import Misc
        >>> round(float(Misc.transmittance_dir(0.1, 2., rot=0.05)), 4)
        0.7408
        '''
        return np.exp(-(rot + aot) * air_mass)

    @staticmethod
    def air_mass(sza, vza):
        r'''Two-way geometric air mass :math:`M = 1/\cos\theta_s + 1/\cos\theta_v` (angles in degrees)

        Examples
        --------
        >>> from hgrs.utils import Misc
        >>> round(float(Misc.air_mass(30, 10)), 4)
        2.1701
        '''
        return 1 / np.cos(np.radians(vza)) + 1 / np.cos(np.radians(sza))

    @staticmethod
    def earth_sun_correction(dayofyear):
        r'''
        Earth-Sun distance correction factor for adjustment of mean solar irradiance (Spencer, 1971)

        .. math::

           \left(\frac{d_0}{d}\right)^2 = 1.00011 + 0.034221 \cos\Theta + 0.00128 \sin\Theta
           + 0.000719 \cos 2\Theta + 0.000077 \sin 2\Theta,
           \qquad \Theta = \frac{2\pi\, \mathrm{DOY}}{365}

        :param dayofyear: day of the year (DOY)
        :return: correction factor, multiplying the mean solar irradiance

        Examples
        --------
        Close to the perihelion (early January) and to the aphelion (early July):

        >>> from hgrs.utils import Misc
        >>> round(float(Misc.earth_sun_correction(3)), 4)
        1.0351
        >>> round(float(Misc.earth_sun_correction(185)), 4)
        0.9666

        References
        ----------
        * Spencer, J. W. (1971). Fourier series representation of the position of the sun. *Search*, 2(5),
          172.
        '''
        theta = 2. * np.pi * dayofyear / 365
        d2 = 1.00011 + 0.034221 * np.cos(theta) + 0.00128 * np.sin(theta) + \
             0.000719 * np.cos(2 * theta) + 0.000077 * np.sin(2 * theta)
        return d2

