'''
Deprecated: the drivers are now in the :py:mod:`hgrs.drivers` subpackage, one class per sensor.
:py:class:`Driver` is kept for the scripts and notebooks written for hGRS 1.1.
'''

import warnings

from .drivers import EnmapDriver, PrismaDriver, TanagerDriver, EmitDriver, HyperionDriver  # noqa: F401 (HyperionDriver, EmitDriver re-exported)


class Driver():
    '''
    Deprecated reader of the L1 products, replaced by the classes of :py:mod:`hgrs.drivers`
    (``read_l1c_enmap`` by ``EnmapDriver().read``, ``read_prisma`` by ``PrismaDriver().read``...).

    :param satellite: ``'enmap'``, ``'prisma'``, ``'tanager'`` or ``'emit'``
    '''

    def __init__(self, satellite='enmap'):
        warnings.warn('hgrs.Driver is deprecated, use the drivers of hgrs.drivers (e.g. EnmapDriver)',
                      DeprecationWarning, stacklevel=2)
        self.satellite = satellite
        self.scene_metadata = None
        self.sensor_description = None

    def _keep_state(self, driver, data):
        # attributes set by the reader (scene_metadata, sensor_description, EnMAP band indices...)
        self.__dict__.update(driver.__dict__)
        return data

    def read_l1c_enmap(self, l1c_path, reflectance_unit=False, drop_vars=False, filter_bad_bands=True):
        driver = EnmapDriver()
        return self._keep_state(driver, driver.read(l1c_path, reflectance_unit=reflectance_unit,
                                                    drop_vars=drop_vars, filter_bad_bands=filter_bad_bands))

    def read_prisma(self, l1c_path, l2c_path, reflectance_unit=True, drop_vars=True, geoproject=True,
                    parallel=False):
        driver = PrismaDriver()
        return self._keep_state(driver, driver.read([l1c_path, l2c_path], geoproject=geoproject,
                                                    reflectance_unit=reflectance_unit, drop_vars=drop_vars,
                                                    parallel=parallel))

    def read_l1c_prisma(self, l1c_path, reflectance_unit=False, drop_vars=False):
        driver = PrismaDriver()
        return self._keep_state(driver, driver.read_l1c(l1c_path, reflectance_unit=reflectance_unit,
                                                        drop_vars=drop_vars))

    def read_l2c_prisma(self, l2c_path):
        driver = PrismaDriver()
        return self._keep_state(driver, driver.read_l2c(l2c_path))

    def read_tanager(self, l1c_path, reflectance_unit=False, drop_vars=False, geoproject=True, parallel=False):
        driver = TanagerDriver()
        return self._keep_state(driver, driver.read(l1c_path, geoproject=geoproject,
                                                    reflectance_unit=reflectance_unit, drop_vars=drop_vars,
                                                    parallel=parallel))

    def read_emit(self, l1b_path, sun_azimuth, sun_elevation, view_azimuth, view_zenith, *,
                  geoproject=False, ortho=False, parallel=False):
        driver = EmitDriver()
        return self._keep_state(driver, driver.read(l1b_path, sun_azimuth=sun_azimuth,
                                                    sun_elevation=sun_elevation, view_azimuth=view_azimuth,
                                                    view_zenith=view_zenith, geoproject=geoproject,
                                                    ortho=ortho, parallel=parallel))
