
from setuptools import setup, find_packages

__package__ = 'hgrs'
__version__ = '1.0.5'

setup(
    name=__package__,
    version=__version__,
    packages=find_packages(exclude=['build']),
    package_data={
        '': ['*.nc', '*.txt', '*.csv', '*.dat'],
        'data':['data/*']

    },
    include_package_data=True,

    url='',
    license='Apache v2',
    authors='T. Harmel',
    author_email='tristan.harmel@magellium.fr; ',
    description='Hyperspectral Glint Removal System: Atmospheric correction for aquatic scenes of hyperspectral (vis-SWIR) satellite images',

    # Dependent packages (distributions)
    install_requires=['numpy', 'scipy', 'pandas', 'xarray','rioxarray',
                      'matplotlib', 'rasterio', 'cartopy',
                      'numba','netcdf4','h5py',
                      'geopandas','affine','shapely','memory_profiler' ],

    entry_points={
        'console_scripts': [
            'hgrs = hgrs.run:main'
        ]}
)
