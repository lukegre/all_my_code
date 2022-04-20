from pkg_resources import DistributionNotFound, get_distribution

# from . import _xarray_stats  # xr.accessors should not be visible
from . import (
    analyse,
    carbsys,
    datasets,
    extremes,
    files,
    munging,
    stats,
    utils,
    viz,
)
from .files.download import download_file
from .munging import date_utils

try:
    __version__ = get_distribution("all_my_code").version
except DistributionNotFound:
    __version__ = "version_undefined"
del get_distribution, DistributionNotFound


data = datasets._amc_Data()
