'''
L1 drivers: one module per sensor, all with the interface of :py:class:`~hgrs.drivers.base.L1Driver`.

The processing chain selects the driver by the sensor name (``Process.execute(sensor=...)``):

.. code-block:: python

   from hgrs.drivers import get_driver

   driver = get_driver('enmap')
   l1c = driver.read('ENMAP01-____L1C-DT0000088121_..._V010502_...')

To add a sensor, write its module with a subclass of :py:class:`~hgrs.drivers.base.L1Driver` and add the
class to ``DRIVERS``.
'''

from .base import L1Driver
from .enmap import EnmapDriver
from .prisma import PrismaDriver
from .tanager import TanagerDriver
from .emit import EmitDriver
from .hyperion import HyperionDriver

DRIVERS = {cls.name: cls for cls in (EnmapDriver, PrismaDriver, EmitDriver, HyperionDriver, TanagerDriver)}


def get_driver(sensor):
    '''
    Return a new driver for a sensor.

    :param sensor: sensor name, one of ``hgrs.drivers.DRIVERS`` (case-insensitive)
    :raises ValueError: if the sensor is not supported
    '''
    try:
        return DRIVERS[sensor.lower()]()
    except KeyError:
        raise ValueError(f'unknown sensor {sensor!r}; available: {", ".join(DRIVERS)}') from None


def driver_for_platform(platform):
    '''Return the driver class of a platform name (``platform`` attribute of the products), or None.'''
    for cls in DRIVERS.values():
        if cls.platform == platform:
            return cls
    return None


__all__ = ['L1Driver', 'DRIVERS', 'get_driver', 'driver_for_platform',
           'EnmapDriver', 'PrismaDriver', 'TanagerDriver', 'EmitDriver', 'HyperionDriver']
