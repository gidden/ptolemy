import ptolemy as pt
import xarray as xr


def make_index_raster(pth, iso, idxkey="NAME_1", adm=1):
    r = pt.Rasterize(like=like)
    r.read_shpf(shpf.format(pth=pth, iso=iso, adm=adm), idxkey=idxkey)
    idxr = r.rasterize(strategy="hybrid", verbose=True)
    encoding = encodings(idxr, zlib=True, complevel=5, dtype="int32")
    idxr.to_netcdf(idxf.format(pth=pth, iso=iso), encoding=encoding)


def test_init():
    like = xr.tutorial.load_dataset("air_temperature").air
    r = pt.Rasterize(like=like)
