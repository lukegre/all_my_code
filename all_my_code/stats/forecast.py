import xarray as xr
import numpy as np
from functools import wraps as _wraps


def trend_and_seasonal_cycle(da, x, groupby_dim='time.month', deg=1):
    """
    Forecast the DataArray based on the trend and seasonal cycle

    The following steps are taken:
        1. calculate the trend
        2. remove the trend and calculate the climatology
        3. forecast the trend and add the climatology

    Parameters
    ----------
    x : datetime-like array
        the coordinates of the new output 
    dim : str [time]
        the dimension along which to predict
    deg : int [1]
        the degree of the polynomial fit by which the trend is calculated

    Returns
    -------
    xr.DataArray : the forecasted array along dimension x
    """
    from xarray import DataArray, polyval
    from . time_series import climatology    
    
    dim0, dim1 = groupby_dim.split('.')
    
    # 1 - calculate the trend
    coef = da.polyfit(dim=dim0, deg=deg).polyfit_coefficients
    trend = xr.polyval(da[dim0], coef)
    trend_forecast = xr.polyval(x, coef)

    # 2 - detrend and calculate the climatology
    detrend = da - trend
    clim = climatology(da, tile=False, groupby_dim=groupby_dim)
    # center the climatology around 0
    clim = clim - clim.mean(dim1)
    # tile the climatology to match x
    clim_forecast = clim.sel(**{dim1: getattr(x[dim0].dt, dim1)}).drop(dim1)

    # 4 - add climatology and trend together
    forecast = clim_forecast + trend_forecast
    if isinstance(da, DataArray):
        forecast = forecast.rename(da.name + "_forecast")

    return forecast


@xr.register_dataarray_accessor('forecast')
@xr.register_dataset_accessor('forecast')
class Forecast(object):
    def __init__(self, xarray_object):
        self._obj = xarray_object
        
    @_wraps(trend_and_seasonal_cycle)
    def trend_and_seasonal_cycle(self, *args, **kwargs):
        return trend_and_seasonal_cycle(self._obj, *args, **kwargs)