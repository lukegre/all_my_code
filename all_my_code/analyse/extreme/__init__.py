from xarray import register_dataarray_accessor as _register_dataarray_accessor
from functools import wraps as _wraps
from .detecting import detect_extremes
from .aggregate import event_based_stats_2d_agg


@_register_dataarray_accessor("extremes")
class Extremes(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @_wraps(detect_extremes)
    def detect(self, **kwargs):
        return detect_extremes(self._obj, **kwargs)

    @_wraps(event_based_stats_2d_agg)
    def event_based_stats_2d_agg(self, **kwargs):
        return event_based_stats_2d_agg(self._obj, **kwargs)

