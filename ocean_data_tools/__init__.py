from pkg_resources import DistributionNotFound, get_distribution

from . import _xarray_stats  # xr.accessors should not be visible
from . import gridding, plotting, sparse
from . import xarray_data_prep as prep

try:
    __version__ = get_distribution("ocean_data_tools").version
except DistributionNotFound:
    __version__ = "version_undefined"
del get_distribution, DistributionNotFound
