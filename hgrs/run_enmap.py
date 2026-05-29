''' Executable to process PRISMA L1 images for aquatic environment

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

import os, sys
from docopt import docopt
import logging
from osgeo import gdal

from . import __package__, __version__
from .hgrs_process import Process


def main():
    args = docopt(__doc__, version=__package__ + '_' + __version__)
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
        outfile = basename.replace('L1_STD_OFFL', lev)
        outfile = outfile.replace('.he5', version+'.nc').rstrip('/')

    odir = args['--odir']
    if odir == './':
        odir = os.getcwd()

    if not os.path.exists(odir):
        os.makedirs(odir)

    outfile = os.path.join(odir, outfile)+'.nc'

    if os.path.exists(outfile) & noclobber:
        print('File ' + outfile + ' already processed; skip!')
        sys.exit()

    logging.info('call grs_process for the following paramater. File:' +
                 l1_path + ', output file:' + outfile +
                 f', cams_file:{cams_file}')

    process_ = Process()
    process_.execute(l1_path,
                     cams_file
                     )

    process_.write_output(outfile)

    return


if __name__ == "__main__":
    main()
