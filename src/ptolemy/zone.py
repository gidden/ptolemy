import numpy as np


class Zones(object):
    def __init__(self, mask):
        self.mask = mask

    # def extrude(self, raster=None, values=None, fill=None):
    #     #        self._read_shp()
    #     #        x = self.mask.copy()

    #     if values is None:
    #         # all non-nans as 1
    #         ret = (~np.ma.masked_invalid(x).mask).astype(int)
    #     else:
    #         # all idxs in values as 1
    #         keep = [int(k) for k, v in self.tags.items() if v in values]
    #         ret = np.in1d(x.ravel(), keep).reshape(x.shape)

    #     where = np.argwhere(ret)
    #     self.y_min, self.x_min = where.min(0)
    #     self.y_max, self.x_max = where.max(0) + 1

    #     if raster is not None:
    #         ret = raster * ret.astype(int)

    #     if fill is not None:
    #         ret[ret == 0] = fill

    #     return ret

    # def crop(self, raster):
    #     if len(raster.shape) > 2:
    #         return raster[:, self.y_min:self.y_max, self.x_min:self.x_max]
    #     else:
    #         return raster[self.y_min:self.y_max, self.x_min:self.x_max]

    # def extrude_1d(self, x, positive=False, sort=False, **kwargs):
    #     x = self.crop(self.extrude(raster=x, **kwargs)).ravel()
    #     if positive:
    #         idx = np.isfinite(x) & (x > 0)
    #         x = x[idx]
    #     if sort:
    #         x = np.sort(x)
    #     return x

    def stats(self, raster, func=np.nansum):
        """Provides zonal statistics given a data raster and an index raster.

        Parameters
        ----------
        raster: str or np.ndarray
            the name of the data raster file or the raster array
        func: python function
            the function to use to summarize the data

        Returns
        -------
        data: dict
            dictionary from idxraster unique values to output
        """
        # mask not yet loaded, assume it was given to us in constructor
        # self._read_shp()

        # if type(raster) is not np.ndarray:
        #     with rio.open(raster) as src:
        #         raster = src.read()[0]
        #         nodata = src.nodata
        #         raster[raster == nodata] = 0
        mask = self.mask
        nodata = -1
        raster = raster.sel(lat=mask.coords["lat"], lon=mask.coords["lon"])
        idxs = np.unique(mask)
        idxs = idxs[(idxs != nodata) & (~np.isnan(idxs))]
        data = {idx: func(raster.values[mask.values == idx]) for idx in idxs}

        # if len(self.mask.shape) == 2:
        # else:
        #     idxs = range(self.mask.shape[2])
        #     assert(idxs)
        #     data = {
        #         idx: func(raster * self.mask[:, :, idx]) for idx in idxs}

        return data
