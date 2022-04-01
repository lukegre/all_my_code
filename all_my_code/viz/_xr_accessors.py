import xarray as xr
from functools import wraps
from .maps import map_subplot
from matplotlib.pyplot import text


@xr.register_dataarray_accessor("map")
class Mapping(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @wraps(map_subplot)
    def __call__(self, **kwargs):
        """Plot 2D data on a map. See map.pcolormesh for all call arguments"""
        return self.pcolormesh(**kwargs)
    
    def _plot(self, plot_func='pcolormesh', **kwargs):
        from .maps import map_subplot, fill_lon_gap
        from numpy import ndim
        from cartopy import crs
        
        da = self._obj
        da = da.squeeze()
        if ndim(da) != 2:
            raise ValueError('Can only plot 2D arrays with maps')
        
        da = da.assign_coords(lon=lambda x: x.lon%360).sortby('lon')
        da = fill_lon_gap(da)
        
        map_kwargs = self._get_map_kwargs(kwargs)
        props = dict(robust=True, **map_subplot(**map_kwargs))
        props.update(kwargs)
        img = getattr(da.plot, plot_func)(**props)
        
        if hasattr(img, 'ax'):
            img.axes = img.ax
        
        self.axes = img.axes
        img.set_title = self._text
        
        return img
    
    @wraps(text)
    def _text(self, s, **props):
        from cartopy import crs
        kwargs = dict(
            ha='center', va='center', zorder=30, weight='bold', size=12,
            transform=crs.PlateCarree())
        kwargs.update(props)
        self.axes.set_title('')
        return self.axes.text(90, 50, s, **kwargs)
    
    @staticmethod
    def _get_map_kwargs(kwargs):
        possible_kwargs = 'pos', 'land_color', 'proj', 'round'
        map_kwargs = {k: v for k, v in kwargs.items() if k in possible_kwargs}
        for k in map_kwargs:
            kwargs.pop(k)
        return map_kwargs
    
    @wraps(map_subplot)
    def contourf(self, **kwargs):
        return self._plot(**kwargs, plot_func='contourf')
    
    @wraps(map_subplot)
    def pcolormesh(self, **kwargs):
        return self._plot(**kwargs, plot_func='pcolormesh')
    
    @wraps(map_subplot)
    def contour(self, **kwargs):
        return self._plot(**kwargs, plot_func='contour')
    