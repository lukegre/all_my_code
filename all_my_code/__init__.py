from pkg_resources import DistributionNotFound, get_distribution

# from . import _xarray_stats  # xr.accessors should not be visible
from . import viz
from . import munging
from . import stats
from . import files
from . import analyse
from . import datasets


try:
    __version__ = get_distribution("all_my_code").version
except DistributionNotFound:
    __version__ = "version_undefined"
del get_distribution, DistributionNotFound


data = datasets._amc_Data()