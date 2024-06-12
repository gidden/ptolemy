import pathlib

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyogrio as pio
import pytest
import xarray as xr
from shapely.geometry import box

import ptolemy as pt


URL = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson"
DATA_PATH = pathlib.Path(__file__).parent / "test_data"
LIKE = xr.tutorial.load_dataset("air_temperature").air.isel(time=0)
LIKE["lon"] = LIKE["lon"] - 360
LIKE_BOX = box(
    LIKE.indexes["lon"][0],
    LIKE.indexes["lat"][0],
    LIKE.indexes["lon"][-1],
    LIKE.indexes["lat"][-1],
)
RASTER_STRATEGIES = ["all_touched", "centroid", "hybrid", "majority", "weighted"]


def _natearth_shapes():
    return pio.read_dataframe(URL).loc[lambda df: df.intersects(LIKE_BOX)]


@pytest.fixture(scope="session")
def natearth_shapes():
    return _natearth_shapes()


@pytest.fixture
def indexraster(natearth_shapes):
    idxr = _do_rasterize("weighted", natearth_shapes)
    idxr_reduced = idxr.sel(iso_a3=["USA", "MEX"])
    return pt.IndexRaster.from_weighted_raster(idxr_reduced, "iso_a3")


@pytest.mark.parametrize("as_file", [True, False])
def test_init(as_file, tmp_path):
    like = LIKE
    if as_file:
        like.to_netcdf(tmp_path / "foo.nc")
        r = pt.Rasterize(like=tmp_path / "foo.nc")
    else:
        r = pt.Rasterize(like=like)
    assert r.shape == like.shape
    assert r.coords["lat"].equals(like["lat"])
    assert r.coords["lon"].equals(like["lon"])


@pytest.mark.parametrize("flatten,exp_size", [(True, 1), (False, 14)])
@pytest.mark.parametrize("idxkey", [None, "iso_a3"])
def test_read_shpf(flatten, exp_size, idxkey, natearth_shapes):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(natearth_shapes, flatten=flatten, idxkey=idxkey)
    assert len(r.geoms_idxs) == exp_size
    assert r.idxkey == idxkey


def _do_rasterize(strategy, shapes=URL):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(shapes, idxkey="iso_a3")
    idxr = r.rasterize(strategy=strategy, verbose=True)
    return idxr


@pytest.mark.parametrize("strategy", RASTER_STRATEGIES)
def test_rasterize(strategy, natearth_shapes):
    obs = _do_rasterize(strategy, natearth_shapes)
    exp = xr.open_dataarray(DATA_PATH / f"{strategy}.nc", mode="r")
    assert obs.equals(exp)
    obs.close()
    exp.close()


def test_df_to_raster_roundtrip(natearth_shapes):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(natearth_shapes, idxkey="iso_a3")
    idxr = r.rasterize(strategy="hybrid", verbose=True)
    df = pd.DataFrame(
        {
            "iso_a3": ["USA"] * 2 + ["MEX"] * 2,
            "year": [2015, 2020] * 2,
            "data": [15, 20, 5, 10],
        }
    )
    idx_map = {v: int(k) for k, v in idxr.attrs.items() if int(k) in np.unique(idxr)}
    ds = pt.df_to_raster(df, idxr, "iso_a3", idx_map, coords=["year"])
    assert ds.sum() == 8195

    obs_df = pt.raster_to_df(ds, idxr, func="max", idx_map=idx_map, idx_dim="iso_a3")
    pdt.assert_frame_equal(
        df.sort_values(by=["iso_a3", "year"]).reset_index(drop=True),
        obs_df.sort_values(by=["iso_a3", "year"]).reset_index(drop=True)[df.columns],
        check_dtype=False,
    )


def test_df_to_weighted_raster_roundtrip(natearth_shapes):
    r = pt.Rasterize(like=LIKE)
    r.read_shpf(natearth_shapes, idxkey="iso_a3")
    idxr = r.rasterize(strategy="weighted", verbose=True)
    df = pd.DataFrame(
        {
            "iso_a3": ["USA"] * 2 + ["MEX"] * 2,
            "year": [2015, 2020] * 2,
            "data": [15, 20, 5, 10],
        }
    )
    ds = pt.df_to_weighted_raster(df, idxr, extra_coords=["year"])
    assert np.isclose(ds.data.sel(year=2015).sum(), 3349.75, rtol=1e-5)

    obs_df = pt.raster_to_df(ds, idxr, func="max")
    pdt.assert_frame_equal(
        df.sort_values(by=["iso_a3", "year"]).reset_index(drop=True),
        obs_df.sort_values(by=["iso_a3", "year"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_cell_area():
    exp = pt.cell_area_from_file(LIKE)
    assert exp.index[0] > exp.index[1]
    assert exp.iloc[0] < exp.iloc[1]
    assert np.isclose(exp.sum(), 1299347051157.3113)


def test_indexraster_gridding(indexraster):
    da = xr.DataArray.from_series(
        pd.DataFrame(
            {
                "iso_a3": ["USA"] * 2 + ["MEX"] * 2,
                "year": [2015, 2020] * 2,
                "data": [15, 20, 5, 10],
            }
        )
        .set_index(["iso_a3", "year"])
        .data
    )

    country_weight = indexraster.aggregate(xr.ones_like(LIKE))
    weighted = da / country_weight
    gridded = indexraster.grid(weighted)
    # Grid cell within US agrees
    assert (gridded.sel(lat=40, lon=-105) - weighted.sel(iso_a3="USA")).max() <= 1e-5

    # Boundary cell between USA and MEX agrees
    blat, blon = 27.5, -97.5
    cell_share = indexraster.boundary.sel(
        lat=blat, lon=blon, iso_a3=[1, 2]
    ).assign_coords(iso_a3=indexraster.index)
    assert (
        gridded.sel(lat=blat, lon=blon) - (cell_share * weighted).sum("iso_a3")
    ).max() <= 1e-5

    # Total agrees
    assert (gridded.sum(["lat", "lon"]) - da.sum("iso_a3")).max() <= 1e-5


def test_indexraster_roundtrip_interior(indexraster):
    da = xr.DataArray.from_series(
        pd.DataFrame(
            {
                "iso_a3": ["USA"] * 2 + ["MEX"] * 2,
                "year": [2015, 2020] * 2,
                "data": [15, 20, 5, 10],
            }
        )
        .set_index(["iso_a3", "year"])
        .data
    )

    # interior only roundtrip agrees
    country_weight = indexraster.aggregate(xr.ones_like(LIKE), interior_only=True)
    weighted = da / country_weight
    gridded = indexraster.grid(weighted)
    obs = indexraster.aggregate(gridded, interior_only=True)
    assert abs(obs - da).max() <= 1e-5


def test_indexraster_serialization(indexraster, tmp_path):
    indexraster.to_netcdf(tmp_path / "idxraster.nc")
    obs = pt.IndexRaster.from_netcdf(tmp_path / "idxraster.nc")

    assert indexraster.index.equals(obs.index)
    assert indexraster.boundary.equals(obs.boundary)
    assert indexraster.indicator.equals(obs.indicator)


def test_indexraster_dissolve(indexraster):
    da = xr.DataArray.from_series(
        pd.DataFrame(
            {
                "iso_a3": ["USA"] * 2 + ["MEX"] * 2,
                "year": [2015, 2020] * 2,
                "data": [15, 20, 15, 20],  # same data in USA and MEX
            }
        )
        .set_index(["iso_a3", "year"])
        .data
    )

    mapping = (
        pd.Series({"MEX": "NAM", "USA": "NAM"})
        .rename_axis(index="iso_a3")
        .rename("region")
    )
    idxregion = indexraster.dissolve(mapping)

    assert idxregion.index.equals(pd.Index(["NAM"], name="region"))

    assert (
        idxregion.grid(da.groupby(xr.DataArray.from_series(mapping)).first())
        - indexraster.grid(da)
    ).max() <= 1e-5


if __name__ == "__main__":
    # save rasterized data for regression
    shapes = _natearth_shapes()
    for strategy in RASTER_STRATEGIES:
        print(f"Working on {strategy}")
        idxr = _do_rasterize(strategy, shapes)
        idxr.to_netcdf(DATA_PATH / f"{strategy}.nc", mode="w")
        idxr.close()
