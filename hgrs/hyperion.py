'''Deprecated: moved to :py:mod:`hgrs.drivers.hyperion`.'''
import sys
import warnings

from .drivers import hyperion as _module

warnings.warn('hgrs.hyperion is deprecated, import hgrs.drivers.hyperion instead', DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _module
