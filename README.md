# hGRS package
## Hyperspectral Glint Removal System for hyperspectral remote sensing of the aquatic environment

# WARNING: this is a working version between collaborators please do not use it as is. Stable version will come soon.

### Prerequisites

If you want to get visualization tool, you first need to install the following libraries (more secure with Anaconda):

```
conda install -c conda-forge hvplot bokeh panel datashader jupyter jupyterlab
```

### Installing

First, clone [the repository](https://github.com/Tristanovsk/hgrs#) and execute the following command in the
local copy:

```
pip install . 
```

Current code rely on the import of xesmf/gdal. TODO: make it an optional dependency, if you want to use the reprojection feature for PRISMA please install xemsf:

```commandline
conda install xesmf gdal
```

## Examples
![example spectra](fig/check_spectra.png)

![example l2c](fig/test_L2C_Garda.png)
![example l1c](fig/test_L1C_Garda_water.png)




