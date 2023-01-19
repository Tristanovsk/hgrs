# prismapy package
## Tool for easy loading of PRISMA L1C and L2C format

### Prerequisites

If you want to get visualization tool, you first need to install the following libraries (more secure with Anaconda):

```
conda install -c conda-forge gdal pyproj rasterio datashader cartopy hvplot geoviews jupyter jupyterlab dask rioxarray
```

### Installing

First, clone [the repository](https://github.com/Tristanovsk/prismapy#) and execute the following command in the
local copy:

```
python setup.py install 
```


## Examples
![example spectra](fig/check_spectra.png)

![example l2c](fig/test_L2C_Garda.png)
![example l1c](fig/test_L1C_Garda_water.png)




