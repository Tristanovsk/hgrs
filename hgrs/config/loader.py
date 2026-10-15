'''
Loading of the hGRS configuration files.

The settings are read from, in increasing order of priority:

1. the default configuration shipped with the package (``hgrs/config/default_config.yml``, or
   ``hgrs/config/old_config.yml`` when :py:data:`hgrs.config.behavior.PREVIOUS_BEHAVIOR` is set),
2. the user configuration file: the file given by the ``HGRS_CONFIG`` environment variable or, if
   not set, ``$XDG_CONFIG_HOME/hgrs/config.yml`` (``~/.config/hgrs/config.yml`` by default),
3. the ``HGRS_DATA_ROOT`` environment variable, which overrides ``path: data_root``.

The user file only needs the keys that differ from the defaults, e.g.::

    path:
      data_root: '/path/to/lut'

Run ``hgrs_config`` to print the configuration in use.
'''

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import yaml

from . import behavior


def default_config_file():
    '''Return the packaged default configuration file selected by the behavior switch.'''
    name = 'old_config.yml' if behavior.PREVIOUS_BEHAVIOR else 'default_config.yml'
    return files(__package__).joinpath(name)


def user_config_file():
    '''Return the path of the user configuration file (which may not exist).'''
    if os.environ.get('HGRS_CONFIG'):
        return Path(os.environ['HGRS_CONFIG']).expanduser()
    config_home = os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config'
    return Path(config_home) / 'hgrs' / 'config.yml'


def _merge(base, update):
    '''Recursively merge the dictionary `update` into a copy of `base`.'''
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config():
    '''Load the hGRS configuration (defaults, then user file, then environment variables).'''
    return _load_config(behavior.PREVIOUS_BEHAVIOR)


@lru_cache(maxsize=None)
def _load_config(previous_behavior):
    # `previous_behavior` is only the cache key, so that switching the behavior reloads the defaults
    with default_config_file().open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}

    user_file = user_config_file()
    if user_file.is_file():
        with open(user_file, 'r', encoding='utf-8') as file:
            config = _merge(config, yaml.safe_load(file) or {})
    elif os.environ.get('HGRS_CONFIG'):
        raise FileNotFoundError(f'HGRS_CONFIG points to a missing file: {user_file}')

    if os.environ.get('HGRS_DATA_ROOT'):
        config = _merge(config, {'path': {'data_root': os.environ['HGRS_DATA_ROOT']}})

    config['path']['data_root'] = os.path.expanduser(config['path']['data_root'])
    return config


def main():
    '''Print the configuration files and the resulting configuration (``hgrs_config`` command).'''
    user_file = user_config_file()
    print(f'default configuration: {default_config_file()}')
    print(f'user configuration:    {user_file} ({"found" if user_file.is_file() else "not found"})')
    print('---')
    print(yaml.safe_dump(load_config(), sort_keys=False), end='')
