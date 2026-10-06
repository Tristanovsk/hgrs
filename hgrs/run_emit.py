"""Run hGRS atmospheric correction for an EMIT L1B_RAD NetCDF granule.

Usage:
  hgrs_emit <l1_path> --cams_file=<file> --sun_azimuth=<angle>
            --sun_elevation=<angle> --view_azimuth=<angle> --view_zenith=<angle>
            [-o <ofile>] [--odir=<odir>] [--no-geoprojection] [--ortho]
  hgrs_emit -h | --help
  hgrs_emit -v | --version

Options:
  -h --help                    Show this help.
  -v --version                 Show version.
  <l1_path>                    EMIT L1B_RAD NetCDF product.
  --cams_file=<file>           CAMS file for the acquisition.
  --sun_azimuth=<angle>        Scene solar azimuth, clockwise from north.
  --sun_elevation=<angle>      Scene solar elevation above horizon.
  --view_azimuth=<angle>       Scene view azimuth, clockwise from north.
  --view_zenith=<angle>        Scene view zenith from nadir.
  -o <ofile>                   Output filename.
  --odir=<odir>                Output directory [default: ./].
  --no-geoprojection           Keep source or GLT pixel grid.
  --ortho                      Use EMIT's supplied GLT grid.

EMIT L1B_RAD does not include these scene angles. The current reader treats
provided angles as scene-constant; obtain them from authoritative metadata.
Example:
python -m hgrs.run_emit "/nethome/lavardma/Documents/data/hyperspectral/EMIT/EMIT_L1B_RAD_001_20260823T090112_2623506_033.nc" --cams_file=/nethome/lavardma/Documents/project/atmospheric/prisma_pipeline/data/cams/20260823/data_sfc.nc --sun-azimuth=338.17 --sun-elevation=56.08 --view-azimuth=141.32 --view-zenith=10.08 --odir="test_data/emit" --no-geoprojection
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

from . import __package__, __version__
from .hgrs_process import Process


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('Usage:')[0].strip())
    parser.add_argument('--version', action='version', version=f'{__package__}_{__version__}')
    parser.add_argument('l1_path', type=Path, help='EMIT L1B_RAD NetCDF product')
    parser.add_argument('--cams_file', required=True, help='CAMS file for acquisition')
    parser.add_argument('--sun-azimuth', type=float, required=True)
    parser.add_argument('--sun-elevation', type=float, required=True)
    parser.add_argument('--view-azimuth', type=float, required=True)
    parser.add_argument('--view-zenith', type=float, required=True)
    parser.add_argument('-o', help='Output filename')
    parser.add_argument('--odir', default='./', help='Output directory')
    parser.add_argument('--no-geoprojection', action='store_true')
    parser.add_argument('--ortho', action='store_true', help="Use EMIT's supplied GLT grid")
    args = parser.parse_args()
    path = args.l1_path
    output = args.o or f'{path.stem}_L2A_hgrs_V{__version__}.nc'
    output_dir = os.getcwd() if args.odir == './' else args.odir
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, output)
    process = Process()
    process.execute(
        str(path), args.cams_file, sensor='emit',
        geoproject=not args.no_geoprojection, ortho=args.ortho,
        sun_azimuth=args.sun_azimuth,
        sun_elevation=args.sun_elevation,
        view_azimuth=args.view_azimuth,
        view_zenith=args.view_zenith,
    )
    process.write_output(output)


if __name__ == '__main__':
    main()
