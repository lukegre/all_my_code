from . import spatial
from . import time_series
from . import smoothen
from . import forecast
from . import seas_cycle
from ..utils import make_xarray_accessor as _make_xarray_accessor


_func_registry = [
    time_series.linregress,
    time_series.slope,
    time_series.climatology,
    time_series.deseasonalise,
    time_series.trend,
    time_series.detrend,
    time_series.rolling_stat_parallel,
    time_series.interannual_variability,
    time_series.time_of_emergence_stdev,
    seas_cycle.seascycl_fit_graven,
    seas_cycle.seascycl_fit_climatology,
    smoothen.lowess,
    time_series.anom,
]

_make_xarray_accessor("time_series", _func_registry, accessor_type="both")
_make_xarray_accessor("stats", _func_registry, accessor_type="both")
