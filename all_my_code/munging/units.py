from pydoc import classname
import xarray as xr
from functools import wraps as _wraps
from ..utils import make_xarray_accessor as _make_xarray_accessor, append_attr

def per_nanosec_to_per_day(da):
    return da * 86400e9


def per_nanosec_to_per_year(da):
    return da * 86400e9 * 365.25


def cm_per_hr_to_meters_per_day(da):
    """Used for gas transfer velocity"""
    return da / 100 * 24


def degK_to_degC(da_degK):
    da_degC = da_degK - 273.15
    attrs = da_degK.attrs
    attrs.update(
        units='degreesC',
        valid_min=-2,
        valid_max=100)
    da_degC = da_degC.assign_attrs(**attrs)

    da_degC = append_attr(da_degC, f"Converted from degK to degC")
    
    return da_degC


_make_xarray_accessor(
    class_name="convert",
    func_list=[
        per_nanosec_to_per_day,
        per_nanosec_to_per_year,
        cm_per_hr_to_meters_per_day,
        degK_to_degC]
    )