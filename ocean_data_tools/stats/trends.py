import xarray as xr
import numpy as np
import joblib


def get_slope(xda, dim='time'):
    if xda.shape[0] <= 2:
        # dummy trend
        return xda[0] * np.nan

    if xda.name is None:
        name = 'slope'
    else:
        name = xda.name + '_slope'
    
    xda = xda.assign_coords(time=lambda x: x.time.dt.year)
    trend = (
        xda.polyfit('time', 1, skipna=False)
        .polyfit_coefficients[0]
        .drop('degree')
        .rename(name)
        .assign_attrs(units='units/year')
    )
    
    return trend

     
def calc_rolling_stat_parallel(da_in, func, window_size=3, n_jobs=36, dim='time'):
    
    roll = da_in.rolling(time=window_size, min_periods=3, center=True)
    
    time = xr.concat([t for t, _ in roll], dim)
    
    if n_jobs > 1:
        pool = joblib.Parallel(n_jobs=n_jobs)
        func = joblib.delayed(func)
        queue = [func(xda) for _, xda in roll]
        trends = pool(queue)
    else:
        trends = [func(xda) for _, xda in roll]
        
    trends = xr.concat(trends, dim).assign_coords(time=time)

    return trends


@xr.register_dataarray_accessor('time_series')
@xr.register_dataset_accessor('time_series')
class TimeSeries(object):
    def __init__(self, xarray_object):
        self._obj = xarray_object
    
    def polyfit(self, dim='time', deg=1):
        """
        A wrapper for polyfit. Has default inputs and removes 
        polyfit_coefficients from the name
        """
        da = self._obj
        coefs = da.polyfit(dim, deg=deg)
        if isinstance(da, xr.DataArray):
            coefs = coefs.polyfit_coefficients
        elif isinstance(da, xr.Dataset):
            names = {k: k.replace('_polyfit_coefficients', '') for k in coefs}
            coefs = coefs.rename(names)
        return coefs
    
    def climatology(self, tile=False, groupby_dim='time.month'):
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
        da = self._obj
        dim0, dim1 = groupby_dim.split('.')
        clim = da.groupby(groupby_dim).mean(dim0)
        
        if tile:
            times = getattr(da[dim0].dt, dim1)
            clim = clim.sel(**{dim1: times}).drop(dim1)
            
        return clim
    
    @staticmethod
    def _predict_with_coefficients(coef, x):
        """takes the output of da.polyfit and predicts along x"""
        x = x.astype(float)
        out = []
        for d in coef.degree:
            out += x**d * coef.sel(degree=d),
        out = xr.concat(out, 'degree').sum('degree')
        
        return out
    
    def trend(self, dim='time', deg=1):
        """
        Calculates the trend over the given dimension
        
        Will mask nans where all nans along the dimension
        
        Parameters
        ----------
        dim : str [time]
            the dimension along which the trend will be calculated
        deg : int [1]
            the order of the polynomial used to fit the data
        
        Returns
        -------
        trend : the trends of the input at the given order
        """
        da = self._obj
        
        coef = self.polyfit(dim=dim, deg=deg)
        
        # use the coeficients to predict along the dimension
        trend = TimeSeries._predict_with_coefficients(coef, da[dim])

        # mask nans where all are nan along a given dimension
        trend = trend.where(da.notnull().any(dim))
        
        return trend

    def forecast(self, x, groupby_dim='time.month', deg=1):
        """
        Forecast the DataArray along the given dimension based on the given time
        
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
        da = self._obj
        dim0, dim1 = groupby_dim.split('.')
        
        # 1 - calculate the trend
        coef = self.polyfit(dim=dim0, deg=deg)
        trend = TimeSeries._predict_with_coefficients(coef, da[dim0])
        trend_forecast = TimeSeries._predict_with_coefficients(coef, x)
        
        # 2 - detrend and calculate the climatology
        detrend = da - trend
        clim = self.climatology(tile=False, groupby_dim=groupby_dim)
        # center the climatology around 0
        clim = clim - clim.mean(dim1)
        # tile the climatology to match x
        clim_forecast = clim.sel(**{dim1: getattr(x[dim0].dt, dim1)}).drop(dim1)
        
        # 4 - add climatology and trend together
        forecast = clim_forecast + trend_forecast
        if isinstance(da, xr.DataArray):
            forecast = forecast.rename(da.name + "_forecast")
        
        return forecast