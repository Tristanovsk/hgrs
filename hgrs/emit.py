'''Deprecated: moved to :py:mod:`hgrs.drivers.emit`.'''
import sys
import warnings

from .drivers import emit as _module

warnings.warn('hgrs.emit is deprecated, import hgrs.drivers.emit instead', DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _module
