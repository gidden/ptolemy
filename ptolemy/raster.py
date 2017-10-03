

def rasterize_majority(geoms_idxs, atrans, shape, nodata, ignore_nodata=False, verbose=False):
    """rasterize shapes such that the shape with the majority of area in a
    cell is assigned to that cell

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
        print('Beginning rasterization')

    a = rio.features.rasterize(
        geoms_idxs,
        out_shape=new_shape,
        transform=new_affine,
        dtype=np.int32,  # could be made an argument in the future!
        fill=nodata,
        all_touched=True)

    if verbose:
        print('Finished rasterization')

    weights = np.ones(a.shape, dtype=np.int)
    if ignore_nodata:
        weights[a == nodata] = 0

    mode = lambda a: _mode(a)[0][0]
    fast_mode = lambda a, weights: np.argmax(np.bincount(a, weights=weights))
    # try to be fast..
    try:
        if verbose:
            print('Beginning fast modal calculation')
        ret = utils.block_apply_2d(
            a, (scale, scale), func=fast_mode, weights=weights)
    except:
        warnings.warn("Could not apply fast mode function, using scipy's mode")
        ret = utils.block_apply_2d(a, (scale, scale), func=mode)

    if verbose:
        print('Process complete')

    return ret


def rasterize_pctcover(geom, atrans, shape):
    # https://github.com/sgoodm/python-rasterstats/blob/cell-weights/src/rasterstats/utils.py#L50
    scale = 10
    new_affine, new_shape = rescale_raster_props(atrans, shape, scale)

    rasterized = rio.features.rasterize(
        [(geom, 1)],
        out_shape=new_shape,
        transform=new_affine,
        fill=0,
        all_touched=True)

    min_dtype = np.min_scalar_type(scale ** 2)
    rv_array = rebin_sum(rasterized, shape, min_dtype)
    return rv_array.astype('float32') / (scale ** 2)


class IndexRaster(object):

    def __init__(self):
        # mask could be an xarray dataset
        # profile and tags could be attributes
        self.profile = None
        self.mask = None
        self.tags = None

    def rasterize(self, strategy=None, normalize_weights=True,
                  flatten=False, idxkey=None, like=None, profile=None,
                  verbose=False):
        """Rasterizes the indicies of the current shapefile.

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
        flatten: bool, optional
            if using `weighted` strategy, flatten all shapes into single feature
        idxkey: string, optional
            a key in properties to use. if provided, only the index, value pair
            will be stored in tags
        like: string, optional
            a filename of a raster whose profile should be used
        profile: dict, optional
            the profile dictionary from a raster object
        verbose: bool, optional
            print out status information during rasterization
        """
        if like is None and profile is None:
            raise ValueError('Either like or profile must be supplied')

        _profile = collections.defaultdict(lambda: None)

        if like is not None:
            self.like = like
            with rio.open(like, 'r') as src:
                profile = src.profile

        # declare up raster properties
        _profile.update(profile)
        shape = (_profile['height'], _profile['width'])
        transform = _profile['affine']
        dtype = _profile['dtype']

        if verbose:
            print('Reading in {}'.format(self.shpf))
        with fio.open(self.shpf, 'r') as n:
            geoms_idxs = tuple((c['geometry'], i) for i, c in enumerate(n))

            if strategy == 'weighted':
                if idxkey is not None:
                    self.tags = [c['properties'][idxkey] for c in n]
                else:
                    self.tags = [c['properties'] for c in n]
            else:
                if idxkey is not None:
                    self.tags = {str(i): c['properties'][idxkey]
                                 for i, c in enumerate(n)}
                else:
                    self.tags = {str(i): c['properties']
                                 for i, c in enumerate(n)}
        if flatten:
            print('Flatting geometries to a single feature')
            geoms = [shapely.geometry.shape(geom)
                     for geom, i in geoms_idxs]
            geoms_idxs = [(shapely.ops.cascaded_union(geoms), 0)]

        if verbose:
            print('Beginning rasterization of {} with the {} strategy'.format(
                self.shpf, strategy))

        nodata = -1
        _profile['nodata'] = -1
        _profile['dtype'] = np.int32
        if strategy in ['all_touched', 'centroid']:
            at = strategy == 'all_touched'
            mask = rio.features.rasterize(
                geoms_idxs, all_touched=at,
                out_shape=shape, transform=transform, fill=nodata,
                dtype=dtype)
        elif strategy in ['majority', 'majority_ignore_nodata']:
             # use one more than the biggest for fast mode calcuation (no
             # negative numbers)
            _nodata = geoms_idxs[-1][1] + 1
            ignore_nodata = strategy == 'majority_ignore_nodata'
            mask = rasterize_majority(geoms_idxs, transform, shape, _nodata,
                                      ignore_nodata=ignore_nodata, verbose=verbose)
            mask[mask == _nodata] = nodata
        elif strategy == 'hybrid':
            # centroid mask
            mask_cent = rio.features.rasterize(
                geoms_idxs, all_touched=False,
                out_shape=shape, transform=transform, fill=nodata,
                dtype=dtype)
            if verbose:
                print('Done with mask 1')

            # all touched mask
            mask_at = rio.features.rasterize(
                geoms_idxs, all_touched=True,
                out_shape=shape, transform=transform, fill=nodata,
                dtype=dtype)
            if verbose:
                print('Done with mask 2')

            # add all border cells not covered by mask_cent to mask_cent
            # Note:
            # must subtract nodata because we are adding to places where mask_cent ==
            # nodata
            mask = mask_cent + \
                np.where((mask_cent == nodata) & (mask_at != nodata),
                         mask_at - nodata, 0)
            if verbose:
                print('Done with mask 3')
        elif strategy == 'weighted':
            nodata = 0
            _profile['nodata'] = nodata
            _profile['dtype'] = np.float32
            mask = np.dstack(rasterize_pctcover(geom, transform, shape)
                             for geom, i in geoms_idxs)
            if normalize_weights:
                # normalize along z-axis to catch coastal cells
                zsum = np.sum(mask, axis=2)
                mask /= zsum[:, :, np.newaxis]
        else:
            raise ValueError('Unknown strategy: {}'.format(strategy))

        self.profile = _profile
        self.mask = mask
