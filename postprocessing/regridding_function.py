import xesmf as xe

s_input_L1C_path = "/home/DATA_EX_IPF/gt_hyperspectral/data/common_data/20240430_data_projection/L1/PRS_L1_STD_OFFL_20220701111705_20220701111710_0001.he5"

def regridding(input_dataset, output_grid_size, d_input_crs=4326):
    """
    Take a PRISMA L1C product in sensor geometry (x,y) as input and 
    return it in a georeferenced geometry (lon,lat).
    
    WARNING : Due to the use of the xESMF package, relying on Fortran, 
    some user warnings like : "UserWarning: Input array is not F_CONTIGUOUS. 
    Will affect performance." may be raised. It is not an issue in our case
    (see https://github.com/JiaweiZhuang/xESMF/issues/25).
    
    :param input_dataset: the product to regrid
    :param output_grid_size: (tuple) output grid size in (lon, lat) format
    :param d_input_crs: (int) code EPSG of the related geolocalisation frame

    :return output_dataset: the regularised product
    """
    # setting lon and lat as coordinates
    input_dataset = input_dataset.set_coords(["lon", "lat"])

    # make the grid that the data will be regridded to
    grid_lons = np.linspace(input_dataset.lon.min().item(), input_dataset.lon.max().item(), output_grid_size[0])
    grid_lats = np.linspace(input_dataset.lat.min().item(), input_dataset.lat.max().item(), output_grid_size[1])
    new_grid = xr.Dataset({'lat':(['lat'],grid_lats), 'lon':(['lon'],grid_lons)})
    
    # use periodic=False if either or both the lat and lon dimensions are not regular
    regridder = xe.Regridder(input_dataset, new_grid, 'bilinear', periodic=False, unmapped_to_nan=True)
    
    # regrid the data
    output_dataset = regridder(input_dataset)
    
    # put the wavelength dependant data lost in the process, back in the dataset
    output_dataset = output_dataset.assign(fwhm=input_dataset.fwhm, F0=input_dataset.F0)

    # adding the CRS
    output_dataset.rio.write_crs(d_input_crs, inplace=True)
    output_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    output_dataset.rio.write_coordinate_system(inplace=True)
    
    return output_dataset

input_dataset = read_L1C_data(s_input_L1C_path)

output_dataset = regridding(input_dataset, output_grid_size=(1000, 1000), d_input_crs=4326)

output_dataset.rio.reproject(dst_crs=3857)