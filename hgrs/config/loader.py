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

Create it with ``hgrs_config --init /path/to/lut``; run ``hgrs_config`` to print the configuration in
use and check the LUTs.
'''

import argparse
import datetime
import os
import sys
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import yaml


def default_config_file():
    '''Return the packaged default configuration file selected by the behavior switch.'''
    from . import behavior
    name = 'old_config.yml' if behavior.PREVIOUS_BEHAVIOR else 'default_config.yml'
    return files(__package__).joinpath(name)


def user_config_file():
    '''Return the path of the user configuration file (which may not exist).'''
    if os.environ.get('HGRS_CONFIG'):
        return Path(os.environ['HGRS_CONFIG']).expanduser()
    config_home = os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config'
    return Path(config_home) / 'hgrs' / 'config.yml'


def read_user_config():
    '''
    Return the settings of the user configuration file, ``{}`` if there is none.

    :raises FileNotFoundError: if ``HGRS_CONFIG`` points to a missing file
    '''
    user_file = user_config_file()
    if user_file.is_file():
        with open(user_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file) or {}
    if os.environ.get('HGRS_CONFIG'):
        raise FileNotFoundError(f'HGRS_CONFIG points to a missing file: {user_file}')
    return {}


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
    from . import behavior
    return _load_config(behavior.PREVIOUS_BEHAVIOR)


@lru_cache(maxsize=None)
def _load_config(previous_behavior):
    # `previous_behavior` is only the cache key, so that switching the behavior reloads the defaults
    with default_config_file().open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}

    config = _merge(config, read_user_config())

    if os.environ.get('HGRS_DATA_ROOT'):
        config = _merge(config, {'path': {'data_root': os.environ['HGRS_DATA_ROOT']}})

    config['path']['data_root'] = os.path.expanduser(config['path']['data_root'])
    return config


def init_user_config(data_root=None, force=False):
    '''
    Write the user configuration file with the LUT folder `data_root` and, commented, the default
    settings (to uncomment and change if needed).

    :param data_root: folder of the LUTs (``path/to/lut`` placeholder if None)
    :param force: overwrite an existing user configuration file
    :return: path of the file written
    :raises FileExistsError: if the file exists and `force` is False
    '''
    user_file = user_config_file()
    if user_file.exists() and not force:
        raise FileExistsError(f'{user_file} already exists (use --force to overwrite it)')

    data_root = str(Path(data_root).expanduser().absolute()) if data_root else 'path/to/lut'
    defaults = files(__package__).joinpath('default_config.yml').read_text(encoding='utf-8')
    defaults = defaults.split('\n\n', 1)[1]  # without the header of the packaged file
    commented = ''.join(('# ' + line if line.strip() else '#\n') if not line.startswith('#') else line
                        for line in defaults.splitlines(keepends=True))
    content = (
        f"# hGRS user configuration, created by `hgrs_config --init` on {datetime.date.today()}.\n"
        "# Only the keys that differ from the packaged defaults are needed: to change a setting, copy\n"
        "# it (with its parent keys) from the defaults below and uncomment it.\n"
        "# Print the configuration in use and check the LUTs with `hgrs_config`.\n"
        "\n"
        "path:\n"
        f"  data_root: '{data_root}'\n"
        "\n"
        "# ------------------------------------------------------------------------------------------\n"
        "# Packaged defaults (hgrs/config/default_config.yml)\n"
        "# ------------------------------------------------------------------------------------------\n"
        f"{commented}"
    )
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text(content, encoding='utf-8')
    return user_file


def lut_status():
    '''
    Return the LUTs of the configuration in use as a list of ``(name, path, exists)``.
    '''
    from .app_config import get_app_config
    config = get_app_config()
    return [(name, path, path.is_file()) for name, path in
            ((name, config.resolve_lut(name)) for name in config.lut_files)]


def main(argv=None):
    '''
    ``hgrs_config`` command: print the configuration in use and check the LUTs (exit code 1 if a LUT
    is missing), or create the user configuration file with ``--init``.
    '''
    parser = argparse.ArgumentParser(
        prog='hgrs_config',
        description='Print the hGRS configuration in use and check the LUTs, '
                    'or create the user configuration file.')
    parser.add_argument('--init', nargs='?', const='', metavar='DATA_ROOT',
                        help='create the user configuration file, with DATA_ROOT the folder of the LUTs')
    parser.add_argument('--force', action='store_true',
                        help='with --init, overwrite an existing user configuration file')
    args = parser.parse_args(argv)

    if args.init is not None:
        try:
            user_file = init_user_config(args.init or None, force=args.force)
        except FileExistsError as error:
            print(f'error: {error}', file=sys.stderr)
            return 1
        print(f'user configuration written: {user_file}')
        if not args.init:
            print('set the folder of the LUTs in it (path: data_root)')
        _load_config.cache_clear()
        print('---')

    from . import behavior
    user_file = user_config_file()
    print(f'default configuration: {default_config_file()}')
    print(f'user configuration:    {user_file} ({"found" if user_file.is_file() else "not found"})')
    print(f'previous behavior:     {behavior.PREVIOUS_BEHAVIOR}')
    print('---')
    print(yaml.safe_dump(load_config(), sort_keys=False), end='')
    print('---')
    missing = False
    for name, path, exists in lut_status():
        print(f'LUT {name:13s} {"ok     " if exists else "MISSING"} {path}')
        missing |= not exists
    if missing:
        print('error: LUT file(s) missing; set the folder of the LUTs with '
              '`hgrs_config --init /path/to/lut` (or path: data_root in the user configuration file, '
              'or HGRS_DATA_ROOT)', file=sys.stderr)
        return 1
    return 0
