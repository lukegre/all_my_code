from os import access
import xarray as xr
import numpy as np
import joblib
from functools import wraps as _wraps
from ..utils import make_xarray_accessor as _make_xarray_accessor


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
    Calculate the first order linear slope per {dim} unit

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
    from warnings import filterwarnings

    filterwarnings('ignore', message=".*poorly conditioned.*")

    # assign the dimension as step from 1 to the length of the dimension
    da = da.assign_coords(**{dim: np.arange(da[dim].size)})
    slope = (
        da.polyfit(dim, 1, skipna=True)
        .polyfit_coefficients[0]
        .drop('degree')
        .assign_attrs(units=f'units/{dim}_step'))
    slope = slope.append_attrs(history=f'Calculated linear slope along the `{dim}` dimension')

    return slope


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

    clim = clim.append_attrs(
        history=f'Calculated the `{dim1}` climatology along the `{dim0}` dimension')

    return clim


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

    deseasonalised = deseasonalised.append_attrs(
        history=f'Removed the `{dim1}` climatology along the `{dim0}` dimension')

    return deseasonalised


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

    trend = trend.append_attrs(
        history=f'Calculated the {deg}-order trend along the `{dim}` dimension')

    return trend


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
    da_detrend = da_detrend.append_attrs(
        history=f'Removed the {deg}-order trend along the `{dim}` dimension')

    return da_detrend


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

    name = getattr(y, 'name', None)
    if drop_polyfit_name:
        rename_dict = {k: k.replace('polyfit_', '') for k in fit}
        fit = fit.rename(rename_dict)
    elif name is not None:
        rename_dict = {k: f"{name}_{k}" for k in fit}
        fit = fit.rename(rename_dict)

    fit = fit.append_attrs(history="Full linear regression with all stats (coefs, r2, pvalue, rmse, +)")

    return fit


def time_of_emergence_stdev(da, deseasonalise=True, noise_multiplier=2, detrend_poly_order=1, dim='time'):
    """
    Calculate time of emergence based on standard deviation 

    Parameters
    ----------
    da : xr.DataArray
        data for which to calculate the ToE
    type : str [deseason | seasonal_cycle]
        determines if the "noise" of the ToE calculation is based on
        the standard deviation of the detrended data or based on the 
        standard deviation of the seasonal cycle
    noise_multiplier : float [2]
        how much confidence you want - sigma * 2 = 95th percentile
    dim : str [time]
        dimension along which ToE is calculated

    Returns
    -------
    time_of_emergence : xr.DataArray
        the time of emergence in years
    """

    name = getattr(da, 'name', None)
    name = name + '_' if name is not None else ''
    
    noise_in = da.time_series.detrend(deg=detrend_poly_order, dim=dim)
    if deseasonalise:
        noise_in = noise_in.time_series.deseasonalise()
        
    noise = noise_in.std(dim) * noise_multiplier
    slope = da.resample(**{dim: '1AS'}).mean().time_series.slope(dim=dim)
        
    toe = (noise / abs(slope)).assign_attrs(
        description=(
            'time of emergence in years (based on standard deviation) '
        ),
        long_name=f'{name}time_of_emergence',
        units='years'
    )
    toe = toe.append_attrs(history="Time of Emergence calculated using standard deviation")
    
    return toe
        

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
    time_of_emergence_stdev,
]


_make_xarray_accessor("time_series", _func_registry, accessor_type='both')
_make_xarray_accessor("ts", _func_registry, accessor_type='both')