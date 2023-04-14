'''
v1.0.0 open and visualize
v1.0.1 orient North of rasters (waiting for proper georeference)
v1.0.2 improvement of solar irradiance data and convolution for reflectance conversion
v1.0.3 development of the process kernel
v1.0.4 reorganizing modules
'''
__version__ = '1.0.4'

from .auxdata import auxdata, solar_irradiance
from .hgrs_kernel import algo, water_vapor, aerosol, product
