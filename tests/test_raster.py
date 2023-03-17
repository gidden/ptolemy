import os
import pathlib

import ptolemy as pt
import pytest
import xarray as xr

URL = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson"
DATA_PATH = pathlib.Path(__file__).parent / "test_data"
LIKE = xr.tutorial.load_dataset("air_temperature").air.isel(time=0)
LIKE["lon"] = LIKE["lon"] - 360
RASTER_STRATEGIES = ["all_touched", "centroid", "hybrid", "majority", "weighted"]


# yes this is silly, but needed until we get a test without lfs
def test_foo():
    assert 42 == 42


@pytest.mark.lfs
@pytest.mark.parametrize("as_file", [True, False])
def test_init(as_file):
    like = LIKE
    if as_file:
        like.to_netcdf(DATA_PATH / "foo.nc")
        r = pt.Rasterize(like=DATA_PATH / "foo.nc")
    else:
        r = pt.Rasterize(like=like)
    assert r.shape == like.shape
    assert r.coords["lat"].equals(like["lat"])
    assert r.coords["lon"].equals(like["lon"])


@pytest.mark.lfs
@pytest.mark.parametrize("flatten,exp_size", [(True, 1), (False, 177)])
@pytest.mark.parametrize("idxkey", [None, "iso_a3"])
def test_read_shpf(flatten, exp_size, idxkey):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(URL, flatten=flatten, idxkey=idxkey)
    assert len(r.geoms_idxs) == exp_size
    assert r.idxkey == idxkey


def _do_rasterize(strategy):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(URL, idxkey="iso_a3")
    idxr = r.rasterize(strategy=strategy, verbose=True)
    return idxr


@pytest.mark.lfs
@pytest.mark.parametrize("strategy", RASTER_STRATEGIES)
def test_rasterize(strategy):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(URL)
    obs = _do_rasterize(strategy)
    exp = xr.open_dataarray(DATA_PATH / f"{strategy}.nc", mode="r")
    assert obs.equals(exp)
    obs.close()
    exp.close()


if __name__ == "__main__":
    # save rasterized data for regression
    for strategy in RASTER_STRATEGIES:
        print(f"Working on {strategy}")
        idxr = _do_rasterize(strategy)
        idxr.to_netcdf(DATA_PATH / f"{strategy}.nc", mode="w")
        idxr.close()
