
import xarray as xr
import numpy as np
import joblib
import os


# SEASONAL CYCLE FITTING # 
def fit_graven2013(da, n_jobs=24):
    """
    Fits a seasonal cycle to data using cos and sin functions. 
    Accounts for annual and 6-monthly cycles. 
    
    Fits the Graven et al. (2013)
    assumes time, lat, lon dataarray where time is a monthly resolution
    
    """
    
    def fit_graven2013_numpy(x, y):
        """
        Fits a seasonal cycle to data using cos and sin functions. 
        Accounts for annual and 6-monthly cycles. 

        Args:
            x (array-like): the time-dimension
            y (array-like): the time-varying value to fit a seasonal cycle to

        Returns:
            array-like: a fitted seasonal cycle (y-hat)
        """
        from scipy import optimize
        from numpy import sin, cos, pi, nan

        def func(t, a1, a2, a3, a4, a5, a6, a7):
            """function to fit as defined by Peter"""
            return (
                a1 + a2*t + a3*t**2 + 
                a4 * sin(2 * pi * t) + 
                a5 * cos(2 * pi * t) + 
                a6 * sin(4 * pi * t) + 
                a7 * cos(4 * pi * t))
        try:
            params, params_covariance = optimize.curve_fit(func, x, y)
        except RuntimeError:
            params = [nan] * 7
        yhat = func(x, *params)

        return yhat
    
    da_stack = da.stack(coords=['lat', 'lon'])

    Ynan = da.stack(coords=['lat', 'lon']).values.T
    Yhat = Ynan * np.nan
    
    nan_mask = np.isnan(Ynan).any(axis=1)
    Y = Ynan[~nan_mask]
    # the x-input has to be in years for the fitting function to work
    x = da.time.values.astype('datetime64[M]').astype(float) / 12
    
    n_yrs = x.size // 12
    lat = da.lat.values
    lon = da.lon.values

    pool = joblib.Parallel(n_jobs=n_jobs)
    func = joblib.delayed(fit_graven2013_numpy)
    queue = [func(x, y) for y in Y]
    Yhat[~nan_mask] = np.array(pool(queue))
    
    Yhat_clim = (
        Yhat
        .reshape(lat.size, lon.size, n_yrs, 12)
        .mean(axis=2)
        .transpose(2, 0, 1))

    djf = Yhat_clim[[-1, 0, 1]].mean(axis=0)
    jja = Yhat_clim[[5, 6, 7]].mean(axis=0)

    rmse = (
        (np.square(Yhat - Ynan).mean(axis=1)**0.5)
        .reshape(lat.size, lon.size))

    dims = 'month', 'lat', 'lon'
    ds = xr.Dataset(coords=dict(month=range(1, 13), lat=da.lat.values, lon=da.lon.values))
    ds['seasonal_clim'] = xr.DataArray(Yhat_clim, dims=dims)
    ds['seas_jja_minus_djf'] = xr.DataArray(jja - djf, dims=dims[1:]).assign_attrs(description='JJA - DFJ')
    ds['rmse'] = xr.DataArray(rmse, dims=dims[1:])

    return ds


def fit_season_min_max(da):
    """
    Simply calculates seasonal cycle amplitude as the clim.max - clim.mean
    
    Args:
        da (xr.DataArray): input array with a time dimension that is mothly
    
    Returns:
        xr.DataArray: A single `timestep` of data
    """
    clim = da.groupby('time.month').mean('time')
    ds = xr.Dataset()
    ds['seasonal_clim'] = clim
    ds['seas_jja_minus_djf'] = (
        clim.sel(month=[6, 7, 8]).mean('month') - 
        clim.sel(month=[12, 1, 2]).mean('month'))
    ds['rmse'] = np.square(da - clim.sel(month=da.time.dt.month)).mean('time')**0.5
    
    return ds


def fit_seasonal_cycle(da, window_size_years=3, func=fit_season_min_max):
    
    amplitudes = []
    y0, y1 = da.time.dt.year[[0, -1]].values
    
    nicename = '.'.join([func.__module__, func.__name__])
    dy = int((window_size_years - 1) / 2)
    years = list(range(y0, y1+1))
    for year in years:
        print(year, end=', ')
        t0 = year - dy
        t1 = year + dy
        s0 = str(t0)
        s1 = str(t1)
        subset = da.sel(time=slice(s0, s1)).dropna('time', how='all')
        amplitudes += func(subset),
    print('')
    
    amplitude = (
        xr.concat(amplitudes, 'time')
        .assign_coords(time=[np.datetime64(f"{y}-01-01") for y in years])
        .assign_attrs(description=f'Seasonal cycle fit with {nicename}')
    )
    
    return amplitude