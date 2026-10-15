"""Process a Planet Tanager basic-radiance HDF5 scene.

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
python -m hgrs.run_tanager   "/nethome/lavardma/Documents/data/hyperspectral/tanager/20250606_090504_75_4001_basic_radiance_hdf5.h5"   --cams_file "/nethome/lavardma/Documents/project/atmospheric/prisma_pipeline/data/cams/20250606/data_sfc.nc"   --odir test_data/tanager   -o 20250606_090504_75_4001_L2A.nc
"""

from __future__ import annotations

import os

from docopt import docopt

from . import __package__, __version__
from .hgrs_process import Process


def main():
    args = docopt(__doc__, version=f'{__package__}_{__version__}')
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


if __name__ == '__main__':
    main()
