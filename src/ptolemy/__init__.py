from importlib.metadata import version as _version

from . import raster
from .raster import Rasterize, df_to_raster, df_to_weighted_raster, raster_to_df, raster_area_from_file

try:
    __version__ = _version("ptolemy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
