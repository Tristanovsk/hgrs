'''
Configuration of hGRS.

The settings are read from, in increasing order of priority:

1. the default configuration shipped with the package (``hgrs/config.yml``),
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

DEFAULT_CONFIG_FILE = files(__package__).joinpath('config.yml')


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


@lru_cache(maxsize=None)
def load_config():
    '''Load the hGRS configuration (defaults, then user file, then environment variables).'''
    with DEFAULT_CONFIG_FILE.open('r') as file:
        config = yaml.safe_load(file)

    user_file = user_config_file()
    if user_file.is_file():
        with open(user_file, 'r') as file:
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
    print(f'default configuration: {DEFAULT_CONFIG_FILE}')
    print(f'user configuration:    {user_file} ({"found" if user_file.is_file() else "not found"})')
    print('---')
    print(yaml.safe_dump(load_config(), sort_keys=False), end='')
