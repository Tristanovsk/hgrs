'''
Compatibility switch between the former and the current behavior of hGRS.

``PREVIOUS_BEHAVIOR`` selects the former parameters (``old_config.yml``) and algorithms when True,
the current ones (``default_config.yml``) when False (default). It is set, in decreasing order of
priority, by:

1. the ``HGRS_PREVIOUS_BEHAVIOR`` environment variable (``1``, ``true``, ``yes`` or ``on``, any case,
   for True; any other value for False),
2. the ``previous_behavior`` key of the user configuration file (see :py:mod:`hgrs.config.loader`).

It is read when ``hgrs`` is imported; in a script or notebook, it can also be changed afterwards::

    from hgrs.config import behavior
    behavior.PREVIOUS_BEHAVIOR = True
'''

import os

from .loader import read_user_config


def _previous_behavior():
    env = os.environ.get('HGRS_PREVIOUS_BEHAVIOR')
    if env is not None:
        return env.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(read_user_config().get('previous_behavior', False))


PREVIOUS_BEHAVIOR = _previous_behavior()
