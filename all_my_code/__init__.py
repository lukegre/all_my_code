from pkg_resources import DistributionNotFound, get_distribution

# from . import _xarray_stats  # xr.accessors should not be visible
from . import viz
from .munging import conform
from . import stats
from . import files

try:
    __version__ = get_distribution("ocean_data_tools").version
except DistributionNotFound:
    __version__ = "version_undefined"
del get_distribution, DistributionNotFound
