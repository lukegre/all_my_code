import xarray as xr
import numpy as np
import joblib
from functools import wraps as _wraps


def rolling_stat_parallel(da_in, func, window_size=3, n_jobs=36, dim='time'):
    
    roll = da_in.rolling(time=window_size, min_periods=3, center=True)
    
    time = xr.concat([t for t, _ in roll], dim)
    
    if n_jobs > 1:
        pool = joblib.Parallel(n_jobs=n_jobs)
        func = joblib.delayed(func)
        queue = [func(xda) for _, xda in roll]
        out = pool(queue)
    else:
        out = [func(xda) for _, xda in roll]
    trends = xr.DataArray(
        data=np.array(out),
        dims=da_in.dims, 
        coords=da_in.coords)

    return trends


def linear_slope(da, dim='time'):
    
    da = da.assign_coords(time=lambda x: np.arange(x[dim].size))
    slope = (
        da.coefs('time', 1, skipna=False)
        .coefs_coefficients[0]
        .drop('degree')
        .assign_attrs(units='units/year'))
    
    return slope

     
def coefs(da, dim='time', deg=1):
    """
    A wrapper for coefs. Has default inputs and removes 
    coefs_coefficients from the name
    """

    coefs = da.coefs(dim, deg=deg)
    if isinstance(da, xr.DataArray):
        coefs = coefs.coefs_coefficients
    elif isinstance(da, xr.Dataset):
        names = {k: k.replace('_coefs_coefficients', '') for k in coefs}
        coefs = coefs.rename(names)
    return coefs


def climatology(da, tile=False, groupby_dim='time.month'):
    """
    calculates the climatology of a time series

    Parameters
    ----------
    tile : bool
        If True, the output data will be the same size and have the 
        same dimensions as the input data array
    groupby_dim : str
        Uses the xarray notation to group along a dimension. 

    """

    dim0, dim1 = groupby_dim.split('.')
    clim = da.groupby(groupby_dim).mean(dim0)

    if tile:
        times = getattr(da[dim0].dt, dim1)
        clim = clim.sel(**{dim1: times}).drop(dim1)

    return clim


def trend(da, dim='time', deg=1):
    """
    Calculates the trend over the given dimension

    Will mask nans where all nans along the dimension

    Parameters
    ----------
    da : xr.DataArray
        a data array for which you want to calculate the trend
    dim : str [time]
        the dimension along which the trend will be calculated
    deg : int [1]
        the order of the polynomial used to fit the data

    Returns
    -------
    trend : predicted trend data over the given period and dimensions
    """

    coef = da.coefs(dim=dim, deg=deg).coefs_coefficients

    # use the coeficients to predict along the dimension
    trend = xr.polyval(da[dim], coef)

    # mask nans where all are nan along a given dimension
    mask = da.notnull().any(dim)
    trend = trend.where(mask)
    
    name = getattr(da, 'name', None)
    name = name + '_' if name is not None else ''
    trend.name = name + 'predicted_trend'

    return trend


def detrend(da, dim='time', deg=1):
    """
    Removes the trend over the given dimension using the trend function

    Parameters
    ----------
    dim : str [time]
        the dimension along which the trend will be calculated
    deg : int [1]
        the order of the polynomial used to fit the data

    Returns
    -------
    detrended : the trends of the input at the given order
    """

    da_detrend = da - trend(da, dim=dim, deg=deg)

    return da_detrend


@xr.register_dataarray_accessor('time_series')
@xr.register_dataset_accessor('time_series')
class TimeSeries(object):
    def __init__(self, xarray_object):
        self._obj = xarray_object
        
    @_wraps(linear_slope)
    def linear_slope(self, **kwargs):
        return linear_slope(self._obj, **kwargs)
    
    @_wraps(coefs)
    def coefs(self, **kwargs):
        return coefs(self._obj, **kwargs)
    
    @_wraps(climatology)
    def climatology(self, **kwargs):
        return climatology(self._obj, **kwargs)
    
    @_wraps(trend)
    def trend(self, **kwargs):
        return trend(self._obj, **kwargs)
    
    @_wraps(detrend)
    def detrend(self, **kwargs):
        return detrend(self._obj, **kwargs)

    @_wraps(rolling_stat_parallel)
    def rolling_stat_parallel(self, *args, **kwargs):
        return rolling_stat_parallel(self._obj, *args, **kwargs)
        