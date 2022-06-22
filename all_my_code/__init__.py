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

import sys
import logging

try:
    __version__ = get_distribution("all_my_code").version
except DistributionNotFound:
    __version__ = "version_undefined"
del get_distribution, DistributionNotFound

# setting up the logging
logger = logging.Logger("AMC", level=logging.WARNING)
while len(logger.handlers) > 0:
    logger.removeHandler(logger.handlers[0])

formatter = logging.Formatter("[%(name)s] %(message)s")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
handler.setLevel(logging.WARNING)
logger.addHandler(handler)
logger.warning(f"version: {__version__}")

data = datasets._amc_Data()
