import xarray as xr
from functools import wraps as _wraps
from . line_plots import plot_ensemble_line_with_std, plot_time_series
from .hovmoller import plot_zonal_anom, plot_zonal_anom_with_trends


@xr.register_dataarray_accessor('viz')
@xr.register_dataset_accessor('viz')
class VizPlots(object):
    def __init__(self, xarray_object):
        self._obj = xarray_object

    @_wraps(plot_ensemble_line_with_std)
    def time_series_ensemble(self, **kwargs):
        return plot_ensemble_line_with_std(self._obj, **kwargs)

    @_wraps(plot_time_series)
    def time_series(self, **kwargs):
        return plot_time_series(self._obj, **kwargs)
        
    def zonal_anomally(self, with_trend=False, lw=0.5, ax=None, **kwargs):
        """
        Plot the zonal anomaly of a 3D xarray object with time, lat, lon dimensions

        Parameters
        ----------
        with_trend : bool
            If True, plot the zonal trends as a third subplot (on the right if ax not specified)
        lw : float
            Linewidth of the contour lines - set to 0 if you don't want contour lines
        ax : list of axes objects
            If None, create a new figure and axes objects. If a list of axes objects, plot on those axes.
            if with_trend is False, then two axes must be given, if True, then three axes objects
            must be given.
        kwargs : dict
            Keyword arguments to be passed to the contourf function

        Returns
        -------
        fig: a figure object
        ax : list of axes objects, but the second object is always a quadmesh object. has colorbar object
        """
        if with_trend:
            return plot_zonal_anom_with_trends(self._obj, ax=ax, lw=lw, **kwargs)
        else:
            return plot_zonal_anom(self._obj, ax=ax, lw=lw, **kwargs)