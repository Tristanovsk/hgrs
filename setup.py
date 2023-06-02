# import ez_setup
# ez_setup.use_setuptools()

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
    license='MIT',
    authors='D. Malik, J. Vicent, T. Harmel',
    author_email='daria.malik@magellium.fr,tristan.harmel@magellium.fr; ',
    description='driver dedicated to the Level 1C and 2C of the ASI PRISMA imagery',

    # Dependent packages (distributions)
    install_requires=['numpy', 'scipy', 'pandas', 'xarray',
                      'matplotlib', 'rasterio', 'osgeo','cartopy',
                      'numba','eoreader',
                      'geopandas','affine','shapely','memory_profiler' ],

    entry_points={
        'console_scripts': [
            'hgrs = TODO'
        ]}
)
