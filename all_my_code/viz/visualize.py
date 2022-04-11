from .line_plots import ensemble_line_with_std, time_series
from .hovmoller import zonal_anomally
from .. utils import make_xarray_accessor as _make_xarray_accessor

_make_xarray_accessor(
    'viz', 
    [ensemble_line_with_std, time_series, zonal_anomally]
)
