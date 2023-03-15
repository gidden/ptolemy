import pathlib

import ptolemy as pt
import pytest
import xarray as xr

URL = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson"
DATA_PATH = pathlib.Path(__file__).parent / "test_data"
LIKE = DATA_PATH / "ssp1_2010_rural.nc"
RASTER_STRATEGIES = ["all_touched", "centroid"]


def make_index_raster(pth, iso, idxkey="NAME_1", adm=1):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(shpf.format(pth=pth, iso=iso, adm=adm), idxkey=idxkey)
    idxr = r.rasterize(strategy="hybrid", verbose=True)
    encoding = encodings(idxr, zlib=True, complevel=5, dtype="int32")
    idxr.to_netcdf(idxf.format(pth=pth, iso=iso), encoding=encoding)


@pytest.mark.parametrize("as_file", [True, False])
def test_init(as_file):
    like = xr.open_dataarray(LIKE)
    if as_file:
        r = pt.Rasterize(like=LIKE)
    else:
        r = pt.Rasterize(like=like)
    assert r.shape == like.shape
    assert r.coords["lat"].equals(like["lat"])
    assert r.coords["lon"].equals(like["lon"])


@pytest.mark.parametrize("flatten,exp_size", [(True, 1), (False, 177)])
@pytest.mark.parametrize("idxkey", [None, "iso_a3"])
def test_read_shpf(flatten, exp_size, idxkey):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(URL, flatten=flatten, idxkey=idxkey)
    assert len(r.geoms_idxs) == exp_size
    assert r.idxkey == idxkey


@pytest.mark.parametrize("strategy", RASTER_STRATEGIES)
def test_rasterize(strategy):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(URL)
    idxr = r.rasterize(strategy=strategy, verbose=True)
    idxr.close()


def _do_rasterize(strategy):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(URL, idxkey="iso_a3")
    idxr = r.rasterize(strategy=strategy, verbose=True)
    return idxr


if __name__ == "__main__":
    # save rasterized data for regression
    for strategy in RASTER_STRATEGIES:
        idxr = _do_rasterize(strategy)
        idxr.to_netcdf(DATA_PATH / f"{strategy}.nc", mode="w")
        idxr.close()
