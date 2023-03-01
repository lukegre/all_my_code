from . import spatial
from . import time_series
from . import outliers
from . import smoothen
from . import forecast
from . import seas_cycle
from . import distributions
from ..utils import make_xarray_accessor as _make_xarray_accessor


_func_registry = [
    time_series.linregress,
    time_series.polyfit,
    time_series.slope,
    time_series.climatology,
    time_series.deseasonalise,
    time_series.trend,
    time_series.detrend,
    time_series.interannual_variability,
    time_series.time_of_emergence_stdev,
    time_series.corr,
    time_series.auto_corr,
    time_series.modes_of_variability,
    seas_cycle.seascycl_fit_graven,
    seas_cycle.seascycl_fit_climatology,
    outliers.mask_outliers_iqr,
    outliers.mask_outliers_std,
    smoothen.lowess,
    smoothen.smooth_monthly,
    time_series.anom,
]

_make_xarray_accessor("time_series", _func_registry, accessor_type="both")
_make_xarray_accessor("stats", _func_registry, accessor_type="both")
