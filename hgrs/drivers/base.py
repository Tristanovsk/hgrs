'''
Common interface of the L1 drivers.

Every driver reads one sensor's L1 product and returns an ``xarray.Dataset`` in the layout expected by
:py:class:`~hgrs.hgrs_process.Process`:

* ``Rtoa`` (TOA reflectance) on ``(wl, y, x)``;
* ``F0`` (solar irradiance convolved with the bands) and ``fwhm`` on ``(wl,)``;
* the angles ``sza``, ``vza``, ``raa`` (degrees) on ``(y, x)``;
* a scalar ``time`` coordinate and the ``L1C_product_name`` attribute.

After :py:meth:`L1Driver.read`, the driver also holds the ``scene_metadata``
(:py:class:`~hgrs.config.SceneMetadata`) and the ``sensor_description``
(:py:class:`~hgrs.config.SensorDescription`) used by the processing chain.
'''

from ..config import SceneMetadata


class L1Driver:
    '''
    Base class of the L1 drivers.

    Subclasses set the class attributes and implement :py:meth:`read`.

    :cvar name: sensor name used to select the driver (``Process.execute(sensor=...)``)
    :cvar platform: platform name written in the products and the scene metadata
    :cvar geometry_args: names of the scene geometry arguments that ``read`` requires (sensors whose L1
        product does not give the angles)
    :cvar tiled_output: write ``Rrs`` with small spatial chunks (one band per chunk) in the L2A file
    '''

    name = None
    platform = None
    geometry_args = ()
    tiled_output = False

    def __init__(self):
        self.scene_metadata = None
        self.sensor_description = None

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
