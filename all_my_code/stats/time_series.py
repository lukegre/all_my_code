import xarray as xr
import numpy as np
import joblib
from functools import wraps as _wraps
from ..utils import add_docs_line1_to_attribute_history, get_unwrapped


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
    """
    Calculate the first order linear slope 

    Parameters
    ----------
    da: xr.DataArray
        the data to calculate the slope of
    dim: str [time]
        the dimension to calculate the slope over
    
    Returns
    -------
    xr.DataArray
        the slope of the data
    """
    
    da = da.assign_coords(time=lambda x: np.arange(x[dim].size))
    slope = (
        da.polyfit('time', 1, skipna=False)
        .polyfit_coefficients[0]
        .drop('degree')
        .assign_attrs(units=f'units/{dim}_step'))
    
    return slope


@add_docs_line1_to_attribute_history
def climatology(da, tile=False, groupby_dim='time.month'):
    """
    Calculate the climatology of a time series

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


@add_docs_line1_to_attribute_history
def deseasonalise(da, groupby_dim='time.month'):
    """
    Remove the seasonal cycle from the time series

    Parameters
    ----------
    da : xr.DataArray
        a data array for which you want to remove the seasonal cycle
    groupby_dim : str
        the dimension to group by

    Returns
    -------
    da_deseasonalised : xr.DataArray
        the time series without the seasonal cycle
    """

    dim0, dim1 = groupby_dim.split('.')
    grp = da.groupby(groupby_dim)
    seasonal_cycle = grp.mean(dim0)
    seasonal_cycle -= seasonal_cycle.mean(dim1)
    deseasonalised = grp - seasonal_cycle
    deseasonalised = deseasonalised.assign_attrs(units=da.attrs.get('units', ''))

    return deseasonalised


@add_docs_line1_to_attribute_history
def trend(da, dim='time', deg=1, coef=None):
    """
    The trend over the given dimension

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


@add_docs_line1_to_attribute_history
def detrend(da, dim='time', deg=1, coef=None):
    """
    Remove the trend along the [time] dimension

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


@add_docs_line1_to_attribute_history
def linregress(y, x=None, dim='time', deg=1, full=True, drop_polyfit_name=True):
    """
    Full linear regression with all stats (coefs, r2, pvalue, rmse, +)

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
    from xarray import DataArray, Dataset

    if isinstance(y, Dataset):
        from ..utils import run_parallel
        inputs = [y[k].rename(k) for k in y]
        outputs = run_parallel(
            linregress, 
            inputs, 
            kwargs=dict(dim=dim, deg=deg, full=full, drop_polyfit_name=False),
            verbose=True)

        return xr.merge(outputs)

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
    elif name := getattr(y, 'name', False):
        rename_dict = {k: f"{name}_{k}" for k in fit}
        fit = fit.rename(rename_dict)

    return fit


@add_docs_line1_to_attribute_history
def interannual_variability(da, dim='time'):
    """
    Calculate the interannual variability of a time series

    Parameters
    ----------
    da : xr.DataArray
        a data array for which you want to calculate the interannual variability

    Returns
    -------
    interannual_variability : xr.DataArray
        the interannual variability of the input
    """

    # calculate the mean of the data
    da_annual = da.resample(time='1AS', loffset='182D').mean()
    da_detrend = da_annual.time_series.detrend()

    # calculate the standard deviation of the data
    interannual_variability = da_detrend.std(dim)

    return interannual_variability


_func_registry = [
    linregress,
    slope,
    climatology,
    deseasonalise,
    trend,
    detrend,
    rolling_stat_parallel,
    interannual_variability,
]


@xr.register_dataarray_accessor('time_series')
@xr.register_dataset_accessor('time_series')
class DataConform(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        for func in _func_registry:
            setattr(self, get_unwrapped(func).__name__, self._make_accessor_func(func))

    def _make_accessor_func(self, func):
        @_wraps(get_unwrapped(func))
        def run_func(*args, **kwargs):
            return func(self._obj, *args, **kwargs)

        return run_func
