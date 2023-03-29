import os
import pathlib

import ptolemy as pt
import pytest
import xarray as xr
import pandas as pd
import numpy as np

URL = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson"
DATA_PATH = pathlib.Path(__file__).parent / "test_data"
LIKE = xr.tutorial.load_dataset("air_temperature").air.isel(time=0)
LIKE["lon"] = LIKE["lon"] - 360
RASTER_STRATEGIES = ["all_touched", "centroid", "hybrid", "majority", "weighted"]


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

@pytest.mark.parametrize("strategy", RASTER_STRATEGIES)
def test_rasterize(strategy):
    obs = _do_rasterize(strategy)
    exp = xr.open_dataarray(DATA_PATH / f"{strategy}.nc", mode="r")
    assert obs.equals(exp)
    obs.close()
    exp.close()

def test_df_to_raster():
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(URL, idxkey="adm0_a3")
    idxr = r.rasterize(strategy='hybrid', verbose=True)
    df = pd.DataFrame({
        "adm0_a3": ['USA'] * 2 + ['MEX'] * 2,
        "year": [2015, 2020] * 2,
        "data": [15, 20, 5, 10],
    })
    idx_map = {v: int(k) for k, v in idxr.attrs.items() if int(k) in np.unique(idxr)}
    ds = pt.df_to_raster(df, idxr, 'adm0_a3', idx_map, coords=['year'])
    assert ds.sum() == 2855

def test_df_to_weighted_raster():
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(URL, idxkey="adm0_a3")
    idxr = r.rasterize(strategy='weighted', verbose=True)
    df = pd.DataFrame({
        "adm0_a3": ['USA'] * 2 + ['MEX'] * 2,
        "year": [2015, 2020] * 2,
        "data": [15, 20, 5, 10],
    })
    ds = pt.df_to_weighted_raster(df, idxr, extra_coords=['year'], sum_dim=['adm0_a3'])
    assert np.isclose(ds.data.sel(year=2015).sum(), 3348.12920833)


def test_cell_area():
    exp = pt.cell_area_from_file(LIKE)
    assert exp.index[0] > exp.index[1]
    assert exp.iloc[0] < exp.iloc[1]
    assert np.isclose(exp.sum(), 1299347051157.3113)


if __name__ == "__main__":
    # save rasterized data for regression
    for strategy in RASTER_STRATEGIES:
        print(f"Working on {strategy}")
        idxr = _do_rasterize(strategy)
        idxr.to_netcdf(DATA_PATH / f"{strategy}.nc", mode="w")
        idxr.close()
