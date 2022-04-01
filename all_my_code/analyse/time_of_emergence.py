import xarray as xr
import pandas as pd
import numpy as np

def calc_trend(poly):
    """
    Calculates the trend given the output from fit_poly.
    
    Parameters
    ----------
    poly: xr.Dataset/Array
        The output from fit_poly function. 
    """
    out = []
    x = poly.time
    poly = poly.drop('time')
    for i in range(poly.degree.size):
        b = poly.sel(degree=i).expand_dims(time=x.size) * x**i
        out += b,
    out = xr.concat(out, 'degree').sum('degree').where(lambda x: x!=0)
    return out

def fit_poly_ds(ds, order=1):
    """
    A wrapper for the polyfit function for xarray. Makes fitting monthly data
    easier and renames the inputs
    """
    
    ds = ds.copy()
    ds['time'] = ds.time.dt.year.values + (ds.time.dt.month - 1) / 12
    
    poly = ds.polyfit('time', order)
    
    poly = poly.rename(**{k: k.replace('_polyfit_coefficients', '') for k in poly})
    poly['time'] = ds.time
    return poly

def corr_lag_ds(ds, lag=1, dim='time'):
    from joblib import Parallel, delayed
    
    keys = list(ds.data_vars.keys())
    proc = Parallel(n_jobs=len(keys))
    func = delayed(xr.corr)
    
    a = ds
    b = ds.shift(**{dim: lag})
    queue = (func(a[k], b[k], dim=dim) for k in keys)
    
    out = xr.merge(proc(queue)).assign_coords(time=ds.time)
    
    return out

def time_of_emergence(ds, mode_of_var='noise', dim='time'):
    """
    Time of Emergence is the time it takes to detect a signal relative to 
    a mode of variability. 
    
    In this function one can calculate the ToE relative to the seasonal cycle 
    and the non-seasonal variability. This done using the approache defined in
    Weatherhead et al. (1998) and used by Lovenduski et al. (2015) and 
    Sutton et al. (2019).
    
    Parameters
    ----------
    ds: xr.Dataset, xr.DataArray
        The data for which you want to calculate the time of emergence on
    mode_of_var: str
        defaults to `noise`, which is the non-seasonal variability. Alternatively,
        one can use `season` which is the seasonal cycle. 
    dim: str
        the name of the time dimension
        
    Returns 
    -------
    xr.Dataset, xr.DataArray
        The output 
    """
    
    poly = fit_poly_ds(ds)
    slope = poly.sel(degree=1)
    trend = calc_trend(poly)
    detrend = ds - trend.assign_coords(time=ds.time)
    
    grp = detrend.groupby(f'{dim}.month')
    if mode_of_var == 'noise':
        mode = grp - grp.mean(dim)
    elif mode_of_var == 'season':
        mode = grp.mean(dim).rename(month=dim)
    elif mode_of_var == 'all':
        mode = detrend
    else:
        raise KeyError('mode_of_var must be `noise` or `season`')
    
    corlag = corr_lag_ds(mode, lag=1, dim='time')
    std = mode.std('time')

    ToE = (
        ((3.3 * std) / abs(slope)) *
        ((1 + corlag) / (1 - corlag))**0.5
    )**(2/3)
    
    ToE = ToE.assign_attrs(
        units='years', 
        standard_name='time_of_emergence',
        description=(
            'Time of Emergence (ToE) calculated using method described in '
            'Weatherhead et al. (1998). '
            'ToE = ((3.3std/slope) / sqrt((1+corlag) / (1-corlag)))**2/3 '
            'where std is the standard deviation of either the noise or seasonal '
            'climatology. The noise is X - seasonal_climatology. Detrending the '
            'X does not make a big difference to the results (12% for Sutton et '
            'al 2018). '),
        mode_of_variability=mode_of_var,
    ).drop(['degree', 'time'])
    
    return ToE
