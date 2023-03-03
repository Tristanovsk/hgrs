# adapted from CRAN R package
import numpy as np
import xarray as xr

def geolocation(raster, lon, lat, fill_gaps=True):
    # TODO current version not working, the method does not seem to be straighforwardly applicable
    # https://www.harrisgeospatial.com/docs/backgroundgltbowtiecorrection.html
    #
    # Estimate the median X and Y coordinates increment from the center column and row
    # from the GLT.
    lon, lat = raster.lon.values, raster.lat.values
    width, height = lon.shape

    psize_x = np.abs(np.nanmedian(np.diff(lon[:, round(height / 2)])))*4
    psize_y = np.abs(np.nanmedian(np.diff(lat[round(width / 2), :])))*4

    # Compute the size of the grid, using the following IDL notation.
    # The CEIL function returns the closest integer greater than or equal to
    # its argument.
    lonmin = np.nanmin(lon)
    lonmax = np.nanmax(lon)
    latmin = np.nanmin(lat)
    latmax = np.nanmax(lat)

    ncols = int(np.ceil((lonmax - lonmin) / psize_x))+1
    nrows = int(np.ceil((latmax - latmin) / psize_y))+1

    # Map all X and Y entries in the GLT to the output grid
    # out_grd <- matrix(data = NA, nrow = nrows, ncol = ncols)
    # vals    <- raster::values(band)

    columns = (np.round((lon - lonmin) / psize_x)).astype(int).flatten()
    rows = (nrows - np.round((lat - latmin) / psize_y)-1).astype(int).flatten()


    xmin = lonmin - 0.5 * psize_x
    xmax = lonmin - 0.5 * psize_x + ncols * psize_x
    ymin = latmax + 0.5 * psize_y - nrows * psize_y
    ymax = latmax + 0.5 * psize_y
    crs = "+init=epsg:4326"

    # transfer values from 1000*1000 cube to the regular 4326 grid ----
    arr = np.full((nrows, ncols), np.nan)
    raster_flat = raster.lat.values.flatten()
    for ipix in range(len(raster_flat)):
        arr[rows[ipix], columns[ipix]] = raster_flat[ipix]


    xarr = xr.Dataset( data_vars=dict(lat=(["y", "x"], arr)),
                       coords=dict(
                           x=np.linspace(xmin,xmax,ncols),
                           y=np.linspace(ymax,ymin,nrows)),
                       )



