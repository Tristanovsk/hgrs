# copyright 2025,  Magellium, J.-P. Burochin

import numpy as np
import xarray as xr

import xesmf as xe
import logging

import numpy as np
import dask.array as da
import xarray as xr
from scipy.interpolate import interp1d

import warnings


class Reproj:
    def __init__(self):
        pass

    @staticmethod
    def regridding(input_dataset, new_grid, parallel=True):
        """
        Take a PRISMA L1C product in sensor geometry (x,y) as input and
        return it in a georeferenced geometry (lon,lat).

        WARNING : Due to the use of the xESMF package, relying on Fortran,
        some user warnings like : "UserWarning: Input array is not F_CONTIGUOUS.
        Will affect performance." may be raised. It is not an issue in our case
        (see https://github.com/JiaweiZhuang/xESMF/issues/25).

        :param input_dataset: the product to regrid
        :param new_grid: grid

        :param d_input_crs: (int) code EPSG of the related geolocalisation frame

        :return output_dataset: the regularised product
        """
        logging.info("georeferencing native image")
        # setting lon and lat as coordinates

        # input_dataset = input_dataset.set_coords(["lon", "lat"])

        # use periodic=False if either or both the lat and lon dimensions are not regular
        regridder = xe.Regridder(
            input_dataset,
            new_grid,
            method="bilinear",
            periodic=False,
            unmapped_to_nan=True,
            parallel=parallel,
        )

        # regrid the data
        output_dataset = regridder(input_dataset)

        output_dataset = output_dataset.assign(fwhm=input_dataset.fwhm)

        return output_dataset


class Misc:
    """
    Miscelaneous utilities
    """

    @staticmethod
    def get_pressure(alt, psl):
        """Compute the pressure for a given altitude
        alt : altitude in meters (float or np.array)
        psl : pressure at sea level in hPa
        palt : pressure at the given altitude in hPa"""

        palt = psl * (1.0 - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    @staticmethod
    def transmittance_dir(aot, air_mass, rot=0):
        return np.exp(-(rot + aot) * air_mass)

    @staticmethod
    def air_mass(sza, vza):
        return 1 / np.cos(np.radians(vza)) + 1 / np.cos(np.radians(sza))

    @staticmethod
    def earth_sun_correction(dayofyear):
        """
        Earth-Sun distance correction factor for adjustment of mean solar irradiance

        :param dayofyear:
        :return: correction factor
        """
        theta = 2.0 * np.pi * dayofyear / 365
        d2 = (
            1.00011
            + 0.034221 * np.cos(theta)
            + 0.00128 * np.sin(theta)
            + 0.000719 * np.cos(2 * theta)
            + 0.000077 * np.sin(2 * theta)
        )
        return d2


def resample(yc, scale):

    dy = yc[1:] - yc[:-1]
    if not np.allclose(dy, dy[0]):
        warnings.warn(
            f"coordinate offset are not the same between pixel (offset/ position): {dy} {yc}"
        )
    dy = np.median(dy)
    N = len(yc) * scale  # /dy #
    sampling_dist = dy / scale
    shift = sampling_dist * (scale - 1) / 2
    y_min = yc[0] - shift
    y_max = y_min + sampling_dist * (N - 1)
    return np.linspace(y_min, y_max, N, endpoint=True)


def interpolate_da(
    a,
    scale=1,
    d=1,
    block_info=None,
):
    # nan_in_input = np.all(~np.isnan(a))
    # if nan_in_input:
    #     warnings.warn("nan in input, will not check if overlap is sufficient")

    # --- Meta case (dask graph construction) ---
    if block_info is None:
        return np.empty((0, 0), dtype=np.float64)

    info = block_info[0]
    loc = info["array-location"]
    chunk_loc = info["chunk-location"]
    ymin, ymax = loc[0]
    xmin, xmax = loc[1]
    number_added_pixels = (chunk_loc[0] * 2 + 1) * d, (chunk_loc[1] * 2 + 1) * d

    # --- Build Grid ---
    NY = ymax - ymin
    NX = xmax - xmin
    yc = ymin + np.arange(NY, dtype=int) - number_added_pixels[0]
    xc = xmin + np.arange(NX, dtype=int) - number_added_pixels[1]
    y = resample(yc.astype(float), scale)
    x = resample(xc.astype(float), scale)

    assert len(yc) > 2
    assert len(xc) > 2

    # --- interpolate ---
    fx = interp1d(xc, a, axis=1, bounds_error=False)
    out = fx(x)

    fy = interp1d(yc, out, axis=0, bounds_error=False)
    out = fy(y)
    # ---- trim ----
    dy = d * scale
    dx = d * scale

    out = out[dy:-dy, dx:-dx]
    # assert not nan_in_input or np.all(~np.isnan(out)), "nan in the array, try to increase overlap"
    return out


def interpolate_xr(
    arr: xr.DataArray,
    xdim="x",
    ydim="y",
    d=1,
    scale=4,
    additional_dims=("wl",),
    chunk_size=32,
):
    def compute_chunk_size(total_length, chunk_size):

        list_chunk = []
        remaining_elems = total_length
        for i in range(total_length // chunk_size):
            list_chunk.append(chunk_size)
            remaining_elems = remaining_elems - chunk_size
        if remaining_elems != 0:

            list_chunk.append(remaining_elems)
        return list_chunk

    src = da.array(arr.transpose(ydim, xdim, *additional_dims).data)
    Ny, Nx = src.shape[0], src.shape[1]
    chunk_sizes = tuple(
        [compute_chunk_size(Ny, chunk_size), compute_chunk_size(Nx, chunk_size)]
        + [-1 for size in src.shape[2:]]
    )
    chunk_sizes_scale = tuple(
        [
            compute_chunk_size(Ny * scale, chunk_size * scale),
            compute_chunk_size(Nx * scale, chunk_size * scale),
        ]
        + [size for size in src.shape[2:]]
    )
    src = src.rechunk(chunk_sizes)
    output = da.map_overlap(
        interpolate_da,
        src,
        scale=scale,
        d=d,
        boundary="reflect",
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
        depth={0: d, 1: d, **{i + 2: 0 for i in range(len(additional_dims))}},
        trim=False,
        align_arrays=False,
        chunks=chunk_sizes_scale,
    )
    coord_add = {k: arr[k] for k in additional_dims}
    coords = {
        ydim: resample(arr[ydim].values, scale),
        xdim: resample(arr[xdim].values, scale),
        **coord_add,
    }
    dims = (ydim, xdim, *additional_dims)
    return xr.DataArray(output, coords=coords, dims=dims)
