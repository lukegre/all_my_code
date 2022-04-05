import xarray as xr
from functools import wraps as _wraps
from . line_plots import plot_ensemble_line_with_std, plot_time_series
from .hovmoller import plot_zonal_anom


@xr.register_dataarray_accessor('viz')
@xr.register_dataset_accessor('viz')
class VizPlots(object):
    def __init__(self, xarray_object):
        self._obj = xarray_object

    @_wraps(plot_ensemble_line_with_std)
    def time_series_ensemble(self, **kwargs):
        return plot_ensemble_line_with_std(self._obj, **kwargs)

    @_wraps(plot_time_series)
    def time_series(self, **kwargs):
        return plot_time_series(self._obj, **kwargs)
        
    @_wraps(plot_zonal_anom)
    def zonal_anomally(self, **kwargs):
        return plot_zonal_anom(self._obj, **kwargs)