'''
Common interface of the L1 drivers.

Every driver reads one sensor's L1 product and returns an ``xarray.Dataset`` in the layout expected by
:py:class:`~hgrs.hgrs_process.Process`:

* ``Rtoa`` (TOA reflectance) on ``(wl, y, x)``;
* ``F0`` (solar irradiance convolved with the bands) and ``fwhm`` on ``(wl,)``; ``F0`` carries the name of
  the solar spectrum and its reference (:py:meth:`L1Driver.solar_irradiance`, :py:meth:`L1Driver.f0_attrs`);
* the angles ``sza``, ``vza``, ``raa`` (degrees) on ``(y, x)``;
* a scalar ``time`` coordinate and the ``L1C_product_name`` attribute.

After :py:meth:`L1Driver.read`, the driver also holds the ``scene_metadata``
(:py:class:`~hgrs.config.SceneMetadata`) and the ``sensor_description``
(:py:class:`~hgrs.config.SensorDescription`) used by the processing chain.
'''

import logging

from ..config import SceneMetadata

logger = logging.getLogger(__name__)


class L1Driver:
    '''
    Base class of the L1 drivers.

    Subclasses set the class attributes and implement :py:meth:`read`.

    :cvar name: sensor name used to select the driver (``Process.execute(sensor=...)``)
    :cvar platform: platform name written in the products and the scene metadata
    :cvar geometry_args: names of the scene geometry arguments that ``read`` requires (sensors whose L1
        product does not give the angles)
    :cvar tiled_output: write ``Rrs`` with small spatial chunks (one band per chunk) in the L2A file
    :cvar mission_solar_spectrum: solar spectrum of the radiometric calibration of the L1 radiance, as
        documented by the mission (name of :py:data:`hgrs.auxdata.SOLAR_SPECTRA`, or a name not available in
        hGRS such as ``'fontenla2011'``); ``None`` when it is not documented
    :cvar radiance_calibration: how the mission calibrates the L1 radiance, with the sources
    :ivar solar_spectrum: solar spectrum converting the L1 radiance into reflectance (name of
        :py:data:`hgrs.auxdata.SOLAR_SPECTRA`); ``None`` (default): the setting
        ``atmospheric_correction: solar_spectrum`` (``tsis`` by default)

    Converting the radiance into reflectance with another solar spectrum than the one of the calibration
    biases ``Rtoa`` by the ratio of the two spectra over the band (e.g. TSIS-1 / Thuillier (2003): +1 to
    +2.5 % in the visible, -3.6 / -4.8 % at 1610 / 2190 nm; Fontenla (2011) / TSIS-1 up to ~6 % below 480 nm).
    Sensors calibrated in reflectance with a sun diffuser embed the solar spectrum of their processing in the
    radiance; sensors calibrated in radiance in the laboratory do not depend on it.
    '''

    name = None
    platform = None
    geometry_args = ()
    tiled_output = False
    mission_solar_spectrum = None
    radiance_calibration = 'not documented in hGRS'
    solar_spectrum = None

    def __init__(self):
        self.scene_metadata = None
        self.sensor_description = None

    def solar_spectrum_name(self):
        '''Name of the solar spectrum used by the driver: ``solar_spectrum``, or the setting.'''
        if self.solar_spectrum is not None:
            return self.solar_spectrum
        from ..config import get_app_config
        return get_app_config().atmospheric_correction.product.solar_spectrum

    def solar_irradiance(self, dayofyear):
        '''
        Solar irradiance spectrum of the driver (:py:meth:`solar_spectrum_name`), corrected for the Earth-Sun
        distance of the day of the year.

        Logs a warning when the mission documents another spectrum for its calibration.

        :param dayofyear: day of the year of the acquisition
        :return: DataArray (wl in nm, mW m-2 nm-1) with the attributes of :py:meth:`f0_attrs`
        '''
        from ..auxdata import SolarIrradiance
        from ..utils import Misc

        name = self.solar_spectrum_name()
        F0 = SolarIrradiance().get(name)
        attrs = self.f0_attrs(name)
        mission = self.mission_solar_spectrum
        if mission is not None and mission != name:
            logger.warning('%s radiance is calibrated with the %s solar spectrum, converted into reflectance '
                           'with %s: Rtoa is biased by their ratio (%s)', self.platform, mission, name,
                           self.radiance_calibration)
        F0 = F0 * Misc.earth_sun_correction(dayofyear)
        F0.attrs = attrs
        return F0

    def f0_attrs(self, name=None):
        '''Attributes of ``F0``: solar spectrum, its reference, and the solar spectrum of the calibration.'''
        from ..auxdata import SOLAR_SPECTRA
        name = self.solar_spectrum_name() if name is None else name
        return {
            'solar_spectrum': name,
            'solar_spectrum_reference': SOLAR_SPECTRA[name],
            'mission_solar_spectrum': str(self.mission_solar_spectrum or 'not documented'),
            'radiance_calibration': self.radiance_calibration,
        }

    def read(self, img_path, *, geoproject=True, **options):
        '''
        Read the L1 product.

        :param img_path: path of the L1 product
        :param geoproject: reproject the image on a map grid (sensors delivered on the swath grid)
        :param options: sensor-specific options, among them the ``geometry_args``
        :return: xarray.Dataset in the layout described in :py:mod:`hgrs.drivers.base`
        '''
        raise NotImplementedError

    @classmethod
    def check_geometry(cls, **geometry):
        '''Raise ``ValueError`` if one of the ``geometry_args`` is missing.'''
        missing = [name for name in cls.geometry_args if geometry.get(name) is None]
        if missing:
            raise ValueError(
                f'{cls.platform} processing requires scene geometry parameters: ' + ', '.join(missing)
            )

    @staticmethod
    def source_georeferencing(dataset):
        '''Capture compact source-grid information before reprojection.'''
        source = {}
        for name in ('lon', 'lat'):
            if name in dataset:
                coordinate = dataset[name]
                source[f'{name}_bounds'] = (
                    float(coordinate.min().values),
                    float(coordinate.max().values),
                )
        try:
            crs = dataset.rio.crs
        except (AttributeError, RuntimeError):
            crs = None
        if crs is not None:
            source['crs'] = str(crs)
        for name in ('x', 'y'):
            if name in dataset.coords and dataset[name].size:
                source[f'{name}_bounds'] = (
                    float(dataset[name].min().values),
                    float(dataset[name].max().values),
                )
        return source

    @staticmethod
    def make_scene_metadata(
        dataset,
        platform,
        source_georeferencing=None,
        source_paths=None,
    ):
        '''Build the :py:class:`~hgrs.config.SceneMetadata` of a dataset read by a driver.'''
        time = dataset['time']
        source_georeferencing = dict(source_georeferencing or {})
        if not source_georeferencing:
            raise ValueError(f'{platform} source georeferencing was not captured')
        raw = {
            'product_name': dataset.attrs.get('L1C_product_name'),
            'source_paths': dict(source_paths or {}),
        }
        return SceneMetadata(
            platform=platform,
            acquisition_time=time,
            solar_zenith=dataset['sza'],
            viewing_zenith=dataset['vza'],
            relative_azimuth=dataset['raa'],
            source_georeferencing=source_georeferencing,
            raw=raw,
        )
