from importlib.metadata import version as _version

from . import raster, zone
from .raster import Rasterize, df_to_raster, update_raster

try:
    __version__ = _version("ptolemy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
