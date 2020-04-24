from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution('glidertools').version
except DistributionNotFound:
    __version__ = 'version_undefined'
del get_distribution, DistributionNotFound

from . import plotting
from . import sparse
from . import gridding
from . import _xarray_stats  # xr.accessors should not be visible
