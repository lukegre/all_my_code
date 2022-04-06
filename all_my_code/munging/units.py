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
