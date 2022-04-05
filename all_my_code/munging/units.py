import xarray as xr
from functools import wraps as _wraps

def per_nanosec_to_day(da):
    return da * 86400e9


def per_nanosec_to_year(da):
    return da * 86400e9 * 365.25


def cm_per_hr_to_meters_per_day(da):
    """Used for gas transfer velocity"""
    return da / 100 * 24


_func_registry = [
    per_nanosec_to_day,
    per_nanosec_to_year,
    cm_per_hr_to_meters_per_day,
]

@xr.register_dataarray_accessor("units")
class Units(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        for func in _func_registry:
            setattr(self, func.__name__, self._make_accessor_func(func))

    def _make_accessor_func(self, func):
        @_wraps(func)
        def run_func(*args, **kwargs):
            return func(self._obj, *args, **kwargs)

        return run_func
