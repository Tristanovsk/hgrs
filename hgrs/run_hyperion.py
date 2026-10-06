"""Run hGRS atmospheric correction for a Hyperion L1R HDF4 scene.

Usage:
  hgrs_hyperion <l1_path> [--cams_file=<file>] [--sun-azimuth=<angle>]
                [--sun-elevation=<angle>] [--satellite-inclination=<angle>]
                [--look-angle=<angle>] [-o <ofile>] [--odir=<odir>]
                [--no-geoprojection]
  hgrs_hyperion -h | --help
  hgrs_hyperion -v | --version

Options:
  -h --help                    Show this help.
  -v --version                 Show version.
  <l1_path>                    Hyperion .L1R file or directory containing it.
  --cams_file=<file>                 Required CAMS file for the acquisition.
  --sun-azimuth=<angle>              Required solar azimuth, degrees clockwise from north.
  --sun-elevation=<angle>            Required solar elevation above the horizon.
  --satellite-inclination=<angle>    Required EO-1 orbital inclination.
  --look-angle=<angle>               Required signed scene look angle from nadir.
  -o <ofile>                   Full or relative output path.
  --odir <odir>                Output directory [default: ./].
  --no-geoprojection           Keep the native pixel grid and lon/lat arrays.
Example:
python -m hgrs.run_hyperion "/nethome/lavardma/Documents/data/hyperspectral/EO-1/EO1H1610692011327110T2_1R/EO1H1610692011327110T2/EO1H1610692011327110T2.L1R" --cams_file="../hgrs/data/cams/20111123/data_sfc.nc" --sun-azimuth=108.360782 --sun-elevation=60.423708 --satellite-inclination=98.14 --look-angle=-7.8964 --no-geoprojection --odir=test_data/hyperion
"""

from __future__ import annotations

import os
from pathlib import Path

from docopt import docopt

from . import __package__, __version__
from .hgrs_process import Process


def main():
    args = docopt(__doc__, version=f'{__package__}_{__version__}')
    l1_path = args['<l1_path>']
    input_path = Path(l1_path)
    if input_path.is_dir():
        candidates = sorted(input_path.rglob('*.L1R'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one .L1R file in {l1_path}; found {len(candidates)}')
        scene_name = candidates[0].stem
    else:
        scene_name = input_path.stem

    outfile = args['-o'] or f'{scene_name}_L2A_hgrs_V{__version__}.nc'
    output_dir = args['--odir']
    if output_dir == './':
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, outfile)

    process = Process()
    process.execute(
        l1_path,
        args['--cams_file'],
        sensor='hyperion',
        geoproject=not args['--no-geoprojection'],
        sun_azimuth=float(args['--sun-azimuth']),
        sun_elevation=float(args['--sun-elevation']),
        satellite_inclination=float(args['--satellite-inclination']),
        look_angle=float(args['--look-angle']),
    )
    process.write_output(outfile)


if __name__ == '__main__':
    main()
