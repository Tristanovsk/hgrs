''' Executable to process PRISMA L1 images for aquatic environment

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

import os, sys
from docopt import docopt
import logging

from . import __package__, __version__
from .hgrs_process import Process


def main():

    args = docopt(__doc__, version=__package__ + '_' + __version__)
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

    logging.info('call grs_process for the following paramater. File:' +
                 l1_path + ', output file:' + outfile +
                 f', cams_file:{cams_file}')

    process_ = Process()
    process_.execute([l1_path,l2_path],
                     cams_file,
                     geoproject=geoproject,
                     )

    process_.write_output(outfile)

    return


if __name__ == "__main__":
    main()
