import itertools
import logging
import warnings

import fiona as fio
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from numpy.lib.stride_tricks import as_strided
from rasterio.features import rasterize as _rasterize
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


logger = logging.getLogger(__name__)


def cell_area_from_file(file, lat_name="lat", lon_name=None):
    """
    Returns the grid cell area by latitude of a raster file.

    Parameters
    ----------
    file : str, pathlib.Path, xr.Dataset, or similar
        a file from which to take transform and latitude objects
    lat_name : str, optional
        the name of the latitude dimension or coordinate
    lon_name : str, optional
        the name of the longitude dimension or coordinate

    Returns
    -------
    area : a geopandas.Series with index of lats and values of area in m^2
    """
    if isinstance(file, (xr.DataArray, xr.Dataset)):
        ds = file
    else:
        ds = xr.open_dataset(file)
    lats = ds[lat_name]
    lons = ds[lon_name] if lon_name else None
    crs = ds.rio.crs if ds.rio.crs else 4326
    return cell_area(lats, lons, crs)


def cell_area(lats, lons=None, crs=4326):
    """
    Computes the grid cell area given centroid latitude and longitude
    coordinates.

    Parameters
    ----------
    lats : array or similar
        latitude coordinates
    lons : array or similar
        longitude coordinates
    crs : string or similar defining CRS, optional
        the origin CRS

    Returns
    -------
    area : a geopandas.Series with index of lats and values of area in m^2
    """
    lat_offset = (lats[1] - lats[0]) / 2
    lon_offset = (lons[1] - lons[0]) / 2 if lons is not None else lat_offset

    lat_pairs = np.nditer([lats - lat_offset, lats + lat_offset])

    return (
        gpd.GeoDataFrame(
            dict(
                geometry=[
                    Polygon(
                        [
                            (-lon_offset, lat[0]),
                            (-lon_offset, lat[1]),
                            (lon_offset, lat[1]),
                            (lon_offset, lat[0]),
                        ]
                    )
                    for lat in lat_pairs
                ],
                lat=lats,
            ),
            crs=crs,
        )
        .set_index("lat")
        .to_crs("Sphere_Cylindrical_Equal_Area")
        .area
    )


def rescale_raster_props(affine, shape, scale):
    """
    Return new transform and shape for a raster for it to be scaled in each
    lat/long dimension by a factor.
    """
    pixel_width = affine[0] / scale
    pixel_height = affine[4] / scale
    topleftlon = affine[2]
    topleftlat = affine[5]

    new_affine = Affine(pixel_width, 0, topleftlon, 0, pixel_height, topleftlat)

    new_shape = (shape[0] * scale, shape[1] * scale)

    return new_affine, new_shape


def block_view_2d(a, blockshape):
    """
    Collapse a 2d array into constituent blocks with a given shape.
    """
    shape = (
        int(a.shape[0] / blockshape[0]),
        int(a.shape[1] / blockshape[1]),
    ) + blockshape
    strides = (blockshape[0] * a.strides[0], blockshape[1] * a.strides[1]) + a.strides
    return as_strided(a, shape=shape, strides=strides)


def block_apply_2d(a, blockshape, func=np.sum, weights=None):
    """
    Apply a function to blocks of an array.

    Returns a reduced array with shape:
    a.shape[0] / blockshape[0], a.shape[1] / blockshape[1]

    TODO: the inner loop can be sped up (likely with cython)

    Parameters
    ----------
    weights: array with same shape as a, optional
        a weight matrix to supply to func (func must take a weights parameter)
    """
    if weights is not None:
        weights = block_view_2d(weights, blockshape)

    blocks = block_view_2d(a, blockshape)
    newshape = int(a.shape[0] / blockshape[0]), int(a.shape[1] / blockshape[1])
    newa = np.empty(newshape, dtype=a.dtype)

    for i, j in itertools.product(range(newshape[0]), range(newshape[1])):
        _weights = weights[i, j, :, :].ravel() if weights is not None else None
        _a = blocks[i, j, :, :].ravel()
        if weights is not None:
            if np.array_equal(_weights, np.zeros(_weights.shape)):
                newa[i, j] = _a[0]  # all weights are 0, just take first value
            else:
                newa[i, j] = func(_a, weights=_weights)

        else:
            newa[i, j] = func(_a)

    return newa


def rasterize_majority(
    geoms_idxs, atrans, shape, nodata, ignore_nodata=False, verbose=False
):
    """
    Rasterize shapes such that the shape with the majority of area in a cell is
    assigned to that cell.

    Parameters
    ----------
    ignore_nodata: bool, optional
         ignore nodata values when determining majority in a cell (appropriate
         for, e.g., when a small island is the only feature in a cell)
    """
    from scipy.stats import mode as _mode

    scale = 10
    new_affine, new_shape = rescale_raster_props(atrans, shape, scale)

    if verbose:
        logger.info("Beginning rasterization")

    a = _rasterize(
        geoms_idxs,
        out_shape=new_shape,
        transform=new_affine,
        dtype=np.int32,  # could be made an argument in the future!
        fill=nodata,
        all_touched=True,
    )

    if verbose:
        logger.info("Finished rasterization")

    weights = np.ones(a.shape, dtype=int)
    if ignore_nodata:
        weights[a == nodata] = 0

    mode = lambda a: _mode(a)[0][0]
    fast_mode = lambda a, weights: np.argmax(np.bincount(a, weights=weights))
    # try to be fast..
    try:
        if verbose:
            logger.info("Beginning fast modal calculation")
        ret = block_apply_2d(a, (scale, scale), func=fast_mode, weights=weights)
    except:  # noqa: 722
        warnings.warn("Could not apply fast mode function, using scipy's mode")
        ret = block_apply_2d(a, (scale, scale), func=mode)

    if verbose:
        logger.info("Process complete")

    return ret


def rebin_sum(a, shape, dtype):
    # https://stackoverflow.com/questions/8090229/
    #   resize-with-averaging-or-rebin-a-numpy-2d-array/8090605#8090605
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).sum(-1, dtype=dtype).sum(1, dtype=dtype)


def rasterize_pctcover(geom, atrans, shape):
    # https://github.com/sgoodm/python-rasterstats/blob/cell-weights/src/rasterstats/utils.py#L50
    scale = 10
    new_affine, new_shape = rescale_raster_props(atrans, shape, scale)

    rasterized = _rasterize(
        [(geom, 1)], out_shape=new_shape, transform=new_affine, fill=0, all_touched=True
    )

    min_dtype = np.min_scalar_type(scale**2)
    rv_array = rebin_sum(rasterized, shape, min_dtype)
    return rv_array.astype("float32") / (scale**2)


def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


class Rasterize:
    """Example use case

    ```
    rs = Rasterize(like='data.nc')
    rs.read_shapefile('boundaries.shp', idxkey='ISO3')
    da = rs.rasterize(strategy='all_touched')
    da.to_netcdf('raster.nc')
    ```
    """

    def __init__(self, shape=None, coords=None, like=None):
        # mask could be an xarray dataset
        # profile and tags could be attributes
        self.tags = None
        self.geoms_idxs = None

        self.nodata = -1
        self.dtype = np.int32

        if like is not None:
            if isinstance(like, (xr.DataArray)):
                da = like
            else:
                da = xr.open_dataarray(like)
            self.shape = da.shape
            self.coords = {
                "lat": da.coords["lat"],
                "lon": da.coords["lon"],
            }
        else:
            self.shape = shape
            self.coords = coords

    def read_shpf(self, shpf, idxkey=None, flatten=None):
        with fio.open(shpf, "r") as n:
            geoms_idxs = tuple((c["geometry"], i) for i, c in enumerate(n))
            self.tags = tuple(
                (str(i), c["properties"][idxkey] if idxkey is not None else "")
                for i, c in enumerate(n)
            )
        if flatten:
            logger.info("Flatting geometries to a single feature")
            geoms = [shape(geom) for geom, i in geoms_idxs]
            geoms_idxs = [(unary_union(geoms), 0)]
        self.geoms_idxs = geoms_idxs
        self.idxkey = idxkey
        return self

    def rasterize(
        self, strategy=None, normalize_weights=True, verbose=False, drop=True
    ):
        """
        Rasterizes the indicies of the current shapefile.

        Parameters
        ----------
        strategy: str, optional
            must be one of
            0. all_touched: GDAL's all_touched = True
            0. centroid: GDAL's all_touched = False
            0. hybrid: a combination of all_touched and centroid, providing a
            better allotment of edge (coastal-like) cells
            0. majority: cells are assigned to the shapes with the majority of
            area within them
            0. weighted: provides a stack of rasters(1 per geometry) of cell
            weights
        normalize_weights: bool, optional
            if using `weighted` strategy, normalize the weights
        verbose: bool, optional
            print out status information during rasterization
        drop: bool, optional
            drop where nodata values in both lat and lon
        """
        if self.geoms_idxs is None:
            raise ValueError("Must call read_shpf() first")

        shape = self.shape
        coords = self.coords
        nodata = self.nodata
        dtype = self.dtype
        geoms_idxs = self.geoms_idxs
        transform = transform_from_latlon(coords["lat"], coords["lon"])
        dims = ["lat", "lon"]

        if verbose:
            logger.info(f"Beginning rasterization with the {strategy} strategy")

        if strategy in ["all_touched", "centroid"]:
            at = strategy == "all_touched"
            mask = _rasterize(
                geoms_idxs,
                all_touched=at,
                out_shape=shape,
                transform=transform,
                fill=nodata,
                dtype=dtype,
            )
        elif strategy in ["majority", "majority_ignore_nodata"]:
            # use one more than the biggest for fast mode calcuation (no
            # negative numbers)
            _nodata = geoms_idxs[-1][1] + 1
            ignore_nodata = strategy == "majority_ignore_nodata"
            mask = rasterize_majority(
                geoms_idxs,
                transform,
                shape,
                _nodata,
                ignore_nodata=ignore_nodata,
                verbose=verbose,
            )
            mask[mask == _nodata] = nodata
        elif strategy == "hybrid":
            # centroid mask
            mask_cent = _rasterize(
                geoms_idxs,
                all_touched=False,
                out_shape=shape,
                transform=transform,
                fill=nodata,
                dtype=dtype,
            )
            if verbose:
                logger.info("Done with mask 1")

            # all touched mask
            mask_at = _rasterize(
                geoms_idxs,
                all_touched=True,
                out_shape=shape,
                transform=transform,
                fill=nodata,
                dtype=dtype,
            )
            if verbose:
                logger.info("Done with mask 2")

            # add all border cells not covered by mask_cent to mask_cent
            # Note:
            # must subtract nodata because we are adding to places where mask_cent ==
            # nodata
            mask = mask_cent + np.where(
                (mask_cent == nodata) & (mask_at != nodata), mask_at - nodata, 0
            )
            if verbose:
                logger.info("Done with mask 3")
        elif strategy == "weighted":
            nodata = 0
            dims += [self.idxkey]
            coords[self.idxkey] = [self.tags[i][1] for geom, i in geoms_idxs]
            mask = xr.DataArray(
                np.dstack(
                    tuple(
                        rasterize_pctcover(geom, transform, shape)
                        for geom, i in geoms_idxs
                    )
                )
            )
            if normalize_weights:
                # normalize along z-axis to catch coastal cells
                zsum = xr.DataArray(np.sum(mask, axis=2))
                mask /= zsum
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        name = "geometry_index"
        da = xr.DataArray(mask, name=name, coords=coords, dims=dims)
        if drop and not "weighted":
            da = da.where(da != nodata, drop=True)
            da.values[np.isnan(da.values)] = nodata
            da = da.astype(dtype)
        #  must be done outside of ctr if drop, not sure why
        da.attrs = self.tags
        return da


def full_like(other, fill_value=np.nan, add_coords={}, replace_vars=[]):
    data = xr.full_like(other, fill_value)
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()
    for k, v in add_coords.items():
        data[k] = v
        data[k].assign_coords()

    if replace_vars:
        data = data.drop(data.data_vars.keys())
        shape = tuple(data.dims.values())
        empty = np.zeros(shape=shape)
        empty[empty == 0] = fill_value
        empty = xr.DataArray(empty, coords=data.coords, dims=data.dims)
        for var in replace_vars:
            data[var] = empty.copy()
    return data


def update_raster(raster, series, idxraster, idx_map):
    """
    Updates a raster array given a raster of indicies and values as columns.

    Parameters
    ----------
    raster : np.ndarray
        the base raster on which to apply updates
    idxrasters : list of np.ndarray or strings
        rasters of indicies where idx_map[series.index] -> idx
    series : list of pd.Series
        values to use to update the raster, the Index of the Series must be the
        keys of the associated idx_map
    """
    if raster.shape != idxraster.shape:
        raise ValueError("Value and Index rasters are not the same shape")
    for validx in series.index:
        # coerce to int because all gdal rasters must have string
        # values. if no idx map is provided, assume map and value
        # indicies match
        mapidx = int(idx_map[validx])
        # this appears to be the fastest way to do replacement in pure
        # python
        replace = idxraster == mapidx
        if not np.any(replace):
            warnings.warn(f"No values found in raster for {mapidx}: {validx}")
        val = series.loc[validx]
        if isinstance(val, pd.Series):
            raise ValueError(f"Multiple entries found for {validx}")
        raster[replace] = val


def df_to_raster(df, idxraster, idx_col, idx_map, ds=None, coords=[], cols=[]):
    """
    Takes data from a pd.DataFrame and deposits it on a raster.

    Parameters
    ----------
    df : pd.DataFrame
        a dataframe with an index aligned with `coords` and data in other columns
    idxraster : xr.DataArray
        an index raster, e.g., from `pt.Rasterize()`
    idx_col : string
        the name of the column linking `df` to `idxraster`
    idx_map : map
        a map of strings to values of `idxraster` to generate the raster
    ds : xr.DataSet(), optional
        a model dataset to use
    coords : list(xr.Coordinate), optional
        coordinates to use to generate a raster
    cols : list(string), optional
        the columns to apply to the raster
    """
    if ds is None:
        cols = cols or sorted(set(df.columns) - set(coords + [idx_col]))
        _coords = {c: sorted(df[c].unique()) for c in coords}
        ds = full_like(idxraster.copy(), add_coords=_coords, replace_vars=cols)

    idxiter = itertools.product(*(sorted(df[c].unique()) for c in coords))
    df = df.set_index(coords)
    for idxs in idxiter:
        sel = dict(zip(coords, idxs))
        data = df.loc[idxs]
        if isinstance(data, pd.Series):  # only one entry, transpose to DF
            data = data.to_frame().T
        data = data.set_index(idx_col)
        for col in cols:
            update_raster(
                ds[col].sel(**sel).values, data[col], idxraster.values, idx_map
            )

    return ds


def df_to_weighted_raster(df, idxraster, col=None, extra_coords=[], sum_dim=None):
    """
    Translates data to a raster with multiple weighting layers. This can be
    used to apply panel data for a series of geometries (e.g., countries) onto
    gridded data.

    Parameters
    ----------
    df : pd.DataFrame
        a dataframe with columns or indicies aligned with coordinates in
        `indexraster`
    idxraster : xr.DataArray
        an index raster with a layer coordinate aligned with `df`. This raster
        can be made with `pt.Rasterize().rasterize()` using the
        `strategy="weighted"` option.
    col : str, optional
        the column in `df` to cast to the map
    extra_coords : list, optional
        additional columns in `df` which should be translated to be coordinates. For
        example, if you want to put panel data onto a raster and that data has a
        "year" column, then you should call this with `coords=["year"]`
    sum_dim : list, optional
        string names of dimension(s) to sum along. This option can be used,
        e.g., to collapse the multiple weighted layers into one 'global' result.
    """
    if len(set(df.index.names) - set(idxraster.dims) - set([None])) == 0:
        # no multi index set, need to align with raster
        idx = list(set(df.columns) & set(idxraster.dims)) + extra_coords
        df = df.set_index(idx)
    if col is not None:
        df = df[col].to_frame()
    data = xr.Dataset.from_dataframe(df)
    if len(data.data_vars) > 1:
        raise ValueError(
            "Currently only support rasterizing one data variable with `df_to_weighted_raster`"
        )
        data = data[list(data.data_vars)][0]  # take only data variable
    result = data * idxraster
    if sum_dim is not None:
        result = result.sum(dim=sum_dim)
    return result


def raster_to_df(
    raster, idxraster, idx_map=None, idx_dim="shape_dim", func="max", drop_zeros=True
):
    """
    Takes data from a raster and makes a pd.DataFrame. Zonal statistics can be
    derived with this function.

    By default, unique values in the index raster areas are returned.

    Parameters
    ----------
    raster : xr.DataArray
        data to make a pd.DataFrame
    idxraster : xr.DataArray
        an index raster, e.g., from `pt.Rasterize()`
    idx_map : dict, optional
        a map of strings to values of `idxraster` if `idxraster` is not weighted
    idx_dim : str, optional
        the name of the index dimension if `idx_map` is provided
    func : string, optional
        a function with can be applied to an array of data. currently supports:
            - max
            - sum
            - mean
    drop_zeros : bool, optional
        drop zeros from the dataframe before returning
    """
    if idx_map:
        idxraster = xr.concat(
            [
                xr.where(idxraster == v, 1, np.nan).expand_dims({idx_dim: [k]}, axis=1)
                for k, v in idx_map.items()
            ],
            dim=idx_dim,
        )

    data = raster * idxraster
    # TODO: there has to be a better way to do this
    dim = ["lat", "lon"]
    if func == "max":
        df = data.max(dim=dim).to_dataframe()
    elif func == "sum":
        df = data.sum(dim=dim).to_dataframe()
    elif func == "mean":
        df = data.mean(dim=dim).to_dataframe()
    if drop_zeros:
        df = df[df != 0].dropna()
    return df.reset_index()
