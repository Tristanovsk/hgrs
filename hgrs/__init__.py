'''
v1.0.0 open and visualize
v1.0.1 orient North of rasters (waiting for proper georeference)
v1.0.2 improvement of solar irradiance data and convolution for reflectance conversion
v1.0.3 development of the process kernel
v1.0.4 reorganizing modules
v1.0.5 investigate solar irradiance reference model
v1.0.6: add reprojection feature
v1.0.7: 2025-11-07 correct bug for Earth-Sun distance, option to remove bad EnMAP bands, refactoring
v1.1.0: new processor for aerosol optical thickness based on non-negativity of retrieved reflectance
v1.1.1: clean up and check which function gaussian, supergaussian is the most suited for spectral integration
v1.1.2: set windows at [1,1] for filter2D (smoothing), need to improve smoothing
'''
__version__ = '1.1.2'

from .utils import Reproj, Misc
from .auxdata import AuxData, SolarIrradiance
from .hgrs_kernel import Algo, WaterVapor, Aerosol, Product, Spectral
from .hgrs_process import Process
from .driver import Driver


import logging

#init logger
logger = logging.getLogger()

level = logging.getLevelName("INFO")
logger.setLevel(level)
