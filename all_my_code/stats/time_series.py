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


def slope(da, dim='time'):
    
    da = da.assign_coords(time=lambda x: np.arange(x[dim].size))
    slope = (
        da.polyfit('time', 1, skipna=False)
        .polyfit_coefficients[0]
        .drop('degree')
        .assign_attrs(units='units/year'))
    
    return slope


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


def trend(da, dim='time', deg=1, coef=None):
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
    coef : xr.DataArray
        the coefficients of the polynomial fit if precalculated
        to save on computation time

    Returns
    -------
    trend : predicted trend data over the given period and dimensions
    """

    if coef is None:
        coef = da.polyfit(dim=dim, deg=deg).polyfit_coefficients

    # use the coeficients to predict along the dimension
    trend = xr.polyval(da[dim], coef)

    # mask nans where all are nan along a given dimension
    mask = da.notnull().any(dim)
    trend = trend.where(mask)
    
    name = getattr(da, 'name', None)
    name = name + '_' if name is not None else ''
    trend.name = name + 'predicted_trend'

    return trend


def detrend(da, dim='time', deg=1, coef=None):
    """
    Removes the trend over the given dimension using the trend function

    Parameters
    ----------
    dim : str [time]
        the dimension along which the trend will be calculated
    deg : int [1]
        the order of the polynomial used to fit the data
    coef : xr.DataArray
        the coefficients of the polynomial fit if precalculated
        to save on computation time

    Returns
    -------
    detrended : the trends of the input at the given order
    """

    da_detrend = da - trend(da, dim=dim, deg=deg, coef=coef)

    return da_detrend


def linregress(y, x=None, dim='time', deg=1, full=True, drop_polyfit_name=True):
    """
    Perform full linear regression with all stats (coefs, r2, pvalue, rmse, +)

    Only slightly slower than using xr.DataArray.polyfit, but returns
    all stats (if full=True)

    Parameters
    ----------
    y : xr.DataArray
        the dependent variable
    x : xr.DataArray
        if None, then dim will be used, otherwise x will be used
    dim : str [time]
        the dimension along which the trend will be calculated
    deg : int [1]
        the order of the polynomial used to fit the data
    full : bool [True]
        if True, the full regression results will be returned

    Returns
    -------
    linregress : xr.DataArray
        the linear regression results containing coefficients, 
        rsquared, pvalue, and rmse 
    """
    from scipy.stats import beta
    from xarray import DataArray

    skipna = True
    # total sum of squares (use this for non-agg dimensions)
    tss = np.square(y - y.mean(dim)).sum(dim)
    
    if x is not None:
        if isinstance(x, DataArray):
            xx = x.munging.drop_0d_coords().dropna(dim)
            xname = x.name
        else:
            xx = DataArray(
                data=x, 
                dims=[dim], 
                coords={dim: y[dim]})
            xname = 'x'
    
        # create regession array
        coords = {k: tss[k].values for k in tss.coords}
        coords[xname] = xx.values
        yy = xr.DataArray(
            data=y.sel(**{dim: xx[dim]}).values, 
            dims=[xname] + list(tss.dims),
            coords=coords)
        dim = xname
    else:
        assert dim in y.dims, 'given dimension is not in y'
        yy = y

    # calculate polyfit
    fit = yy.polyfit(dim, deg, full=full)
    
    if not full:
        return fit
    
    # residual sum of squares
    rss = fit.polyfit_residuals
    
    fit['polyfit_rsquared'] = (1 - (rss / tss)).assign_attrs(description='pearsons r-value squared')
    r = fit['polyfit_rsquared']**0.5

    n = yy[dim].size
    # creating the distribution for pvalue
    dist = beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
    
    # calculating R value
    fit['polyfit_pvalue'] = (r * np.nan).fillna(2 * dist.cdf(-abs(r)))
    fit['polyfit_rmse'] = (rss / n)**0.5

    if drop_polyfit_name:
        rename_dict = {k: k.replace('polyfit_', '') for k in fit}
        fit = fit.rename(rename_dict)

    return fit


@xr.register_dataarray_accessor('time_series')
@xr.register_dataset_accessor('time_series')
class TimeSeries(object):
    def __init__(self, xarray_object):
        self._obj = xarray_object

    @_wraps(linregress)
    def linregress(self, **kwargs):
        return linregress(self._obj, **kwargs)
        
    @_wraps(slope)
    def slope(self, **kwargs):
        return slope(self._obj, **kwargs)
    
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
        