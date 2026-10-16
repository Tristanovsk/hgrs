'''
Command-line tools of hGRS, one function per sensor: ``hgrs_enmap``, ``hgrs_prisma``, ``hgrs_emit``,
``hgrs_hyperion`` and ``hgrs_tanager`` (entry points of ``pyproject.toml``). The options of each command
are given by its ``*_USAGE`` text, printed by ``--help``.
'''

from __future__ import annotations

import os, sys
import argparse
import logging
from pathlib import Path

from docopt import docopt

from . import __package__, __version__
from .hgrs_process import Process

logger = logging.getLogger(__name__)

ENMAP_USAGE = ''' Executable to process EnMAP L1C images for aquatic environment

Usage:
  hgrs_enmap <l1_path>  [--cams_file file] [-o <ofile>] [--odir <odir>]\
 [--opac_model name] [--levname <lev>] [--no_clobber]
  hgrs_enmap -h | --help
  hgrs_enmap -v | --version

Options:
  -h --help        Show this screen.
  -v --version     Show version.

  <l1_path>     Input file to be processed

  --cams_file file  Absolute path of the CAMS file to be used

  -o ofile          Full (absolute or relative) path to output L2 image.
  --odir odir       Ouput directory [default: ./]
  --levname lev     Level naming used for output product [default: L2A_hgrs]
  --no_clobber      Do not process <input_file> if <output_file> already exists.
  --opac_model name  Force the aerosol model (OPAC) to be 'name'
                     (choice: ['ARCT_rh70', 'COAV_rh70', 'DESE_rh70',
                     'MACL_rh70', 'URBA_rh70']) (WARNING: not implemented yet)


  Example:
      L1_path=

      CAMS_path=
      hgrs_enmap $L1_path --cams_file $CAMS_path

'''

PRISMA_USAGE = ''' Executable to process PRISMA L1 images for aquatic environment

Usage:
  hgrs_prisma <l1_path> [--l2_path <l2_path>] [--cams_file <file>] [-o <ofile>] [--odir <odir>] [--levname <lev>] [--no_clobber] [--no-geoprojection]
  hgrs_prisma -h | --help
  hgrs_prisma -v | --version

Options:
  -h --help        Show this screen.
  -v --version     Show version.

  <l1_path>     Input file to be processed

  --l2_path l2_path  Path for L2C PRISMA image to load observation angles,
                     if not provided it is retrieved within the directory
                     of the L1 input image
  --cams_file file   Absolute path of the CAMS file to be used

  -o ofile           Full (absolute or relative) path to output L2 image.
  --odir odir        Ouput directory [default: ./]
  --levname lev      Level naming used for output product [default: L2A_hgrs]
  --no_clobber       Do not process an input file if the output file already exists.
  --no-geoprojection  Keep the PRISMA L2A output on its native pixel grid and
                     retain its longitude and latitude arrays for later projection.

  Example:
      L1_path=
      L2_path=
      CAMS_path=
      hgrs_prisma $L1_path --l2_path $L2_path --cams_file $CAMS_path
'''

EMIT_USAGE = '''Run hGRS atmospheric correction for an EMIT L1B_RAD NetCDF granule.

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
hgrs_emit "/nethome/lavardma/Documents/data/hyperspectral/EMIT/EMIT_L1B_RAD_001_20260823T090112_2623506_033.nc" --cams_file=/nethome/lavardma/Documents/project/atmospheric/prisma_pipeline/data/cams/20260823/data_sfc.nc --sun-azimuth=338.17 --sun-elevation=56.08 --view-azimuth=141.32 --view-zenith=10.08 --odir="test_data/emit" --no-geoprojection
'''

HYPERION_USAGE = '''Run hGRS atmospheric correction for a Hyperion L1R HDF4 scene.

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
hgrs_hyperion "/nethome/lavardma/Documents/data/hyperspectral/EO-1/EO1H1610692011327110T2_1R/EO1H1610692011327110T2/EO1H1610692011327110T2.L1R" --cams_file="../hgrs/data/cams/20111123/data_sfc.nc" --sun-azimuth=108.360782 --sun-elevation=60.423708 --satellite-inclination=98.14 --look-angle=-7.8964 --no-geoprojection --odir=test_data/hyperion
'''

TANAGER_USAGE = '''Process a Planet Tanager basic-radiance HDF5 scene.

Usage:
  hgrs_tanager <l1_path> [--cams_file file] [-o <ofile>] [--odir <odir>]
  hgrs_tanager -h | --help
  hgrs_tanager -v | --version

Options:
  -h --help         Show this screen.
  -v --version      Show version.
  <l1_path>         Input Tanager basic-radiance HDF5 file.
  --cams_file file  CAMS file for the acquisition.
  -o ofile          Full or relative output path.
  --odir odir       Output directory [default: ./].

Exemple:
hgrs_tanager "/nethome/lavardma/Documents/data/hyperspectral/tanager/20250606_090504_75_4001_basic_radiance_hdf5.h5"   --cams_file "/nethome/lavardma/Documents/project/atmospheric/prisma_pipeline/data/cams/20250606/data_sfc.nc"   --odir test_data/tanager   -o 20250606_090504_75_4001_L2A.nc
'''


def enmap():
    '''``hgrs_enmap`` command: process an EnMAP L1C image (options: ``hgrs_enmap --help``).'''
    logging.basicConfig(level=logging.INFO)
    args = docopt(ENMAP_USAGE, version=__package__ + '_' + __version__)
    print(args)
    version = 'V' + __version__
    l1_path = args['<l1_path>']

    cams_file = args['--cams_file']
    noclobber = args['--no_clobber']
    opac_model = args['--opac_model']
    lev = args['--levname']

    ##################################
    # File naming convention
    ##################################
    basename = os.path.basename(l1_path)

    outfile = args['-o']
    if outfile == None:
        outfile = basename.replace('L1C', lev)
        outfile = outfile+ '_'+version+'.nc'

    odir = args['--odir']
    if odir == './':
        odir = os.getcwd()

    if not os.path.exists(odir):
        os.makedirs(odir)

    outfile = os.path.join(odir, outfile)

    if os.path.exists(outfile) & noclobber:
        print('File ' + outfile + ' already processed; skip!')
        sys.exit()

    logger.info('call hgrs_process for the following parameters. File:' +
                l1_path + ', output file:' + outfile +
                f', cams_file:{cams_file}')

    process_ = Process()
    process_.execute(l1_path,
                     cams_file
                     )

    process_.write_output(outfile)

    return


def prisma():
    '''``hgrs_prisma`` command: process a PRISMA L1 image, with the angles of the L2C product (options: ``hgrs_prisma --help``).'''

    logging.basicConfig(level=logging.INFO)
    args = docopt(PRISMA_USAGE, version=__package__ + '_' + __version__)
    print(args)

    version = 'V'+__version__
    l1_path = args['<l1_path>']
    l2_path = args['--l2_path']
    cams_file = args['--cams_file']
    noclobber = args['--no_clobber']
    lev = args['--levname']
    geoproject = not args['--no-geoprojection']

    ##################################
    # File naming convention
    ##################################
    basename = os.path.basename(l1_path)
    idir = os.path.dirname(l1_path)
    if l2_path == None:
        l2_path = l1_path.replace('L1_STD_OFFL', 'L2C_STD')
        #l2_path = opj(idir, l2c)

    outfile = args['-o']
    if outfile == None:
        outfile = basename.replace('L1_STD_OFFL', lev)
        outfile = outfile.replace('.he5', '_'+version+'.nc').rstrip('/')

    odir = args['--odir']
    if odir == './':
        odir = os.getcwd()

    if not os.path.exists(odir):
        os.makedirs(odir)

    outfile = os.path.join(odir, outfile)

    if os.path.exists(outfile) & noclobber:
        print('File ' + outfile + ' already processed; skip!')
        sys.exit()

    logger.info('call hgrs_process for the following parameters. File:' +
                l1_path + ', output file:' + outfile +
                f', cams_file:{cams_file}')

    process_ = Process()
    process_.execute([l1_path,l2_path],
                     cams_file,
                     geoproject=geoproject,
                     )

    process_.write_output(outfile)

    return


def emit():
    '''``hgrs_emit`` command: process an EMIT L1B_RAD granule, with the scene angles given as options
    (options: ``hgrs_emit --help``).'''
    parser = argparse.ArgumentParser(description=EMIT_USAGE.split('Usage:')[0].strip())
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


def hyperion():
    '''``hgrs_hyperion`` command: process a Hyperion L1R scene (options: ``hgrs_hyperion --help``).'''
    args = docopt(HYPERION_USAGE, version=f'{__package__}_{__version__}')
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


def tanager():
    '''``hgrs_tanager`` command: process a Tanager basic-radiance scene (options: ``hgrs_tanager --help``).'''
    args = docopt(TANAGER_USAGE, version=f'{__package__}_{__version__}')
    l1_path = args['<l1_path>']
    outfile = args['-o']
    if outfile is None:
        outfile = os.path.basename(l1_path).replace(
            'basic_radiance_hdf5.h5', f'L2A_hgrs_V{__version__}.nc'
        )
    odir = args['--odir']
    if odir == './':
        odir = os.getcwd()
    os.makedirs(odir, exist_ok=True)
    outfile = os.path.join(odir, outfile)

    process = Process()
    process.execute(l1_path, args['--cams_file'], sensor='tanager')
    process.write_output(outfile)
