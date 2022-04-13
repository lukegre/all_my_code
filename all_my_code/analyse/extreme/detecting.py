import xarray as xr
import numpy as np


def fixed_baseline(da, quantile=0.95, period=slice('1985', '2014'), clim_agg_func='mean', **kwargs):
    
    baseline = da.sel(time=period)

    grp = baseline.groupby('time.month')
    thresh = grp.quantile(quantile, 'time').sel(month=da.time.dt.month)
    clim = getattr(grp, clim_agg_func)('time').sel(month=da.time.dt.month)
    attrs = dict(
        baseline_type='fixed',
        baseline_period=f"{period.start}:{period.stop}",
        threshold_quantile=quantile)
    
    ds = xr.Dataset()
    ds['threshold'] = thresh.assign_attrs(attrs)
    ds['climatology'] = clim.assign_attrs(
        aggregation_function=clim_agg_func, **attrs)
    
    ds = ds.assign_attrs(attrs)
    
    return ds


def detrend_baseline(da, order=1, quantile=0.95, clim_agg_func='mean', **kwargs):
    from stats import trend_poly
    
    trend = da.time_series.trend(deg=order)
    baseline = da - trend
    grp = baseline.groupby('time.month')
    thresh = grp.quantile(quantile, 'time').sel(month=da.time.dt.month)
    clim = getattr(grp, clim_agg_func)('time').sel(month=da.time.dt.month)
    
    attrs = dict(
        baseline_type='shifting',
        baseline_poly_order=order,
        baseline_period=f"{da.time.dt.year.values[0]}:{da.time.dt.year.values[-1]}",
        threshold_quantile=quantile)
    
    ds = xr.Dataset()
    ds['threshold'] = (thresh + trend).assign_attrs(**attrs)
    ds['climatology'] = (clim + trend).assign_attrs(func=clim_agg_func, **attrs)
    ds = ds.assign_attrs(attrs)
    
    return ds


def detect_extremes(
    da, 
    baseline_type='fixed', 
    quantile=0.95, 
    baseline_period=slice('1985', '2014'), 
    verbose=True, 
    n_largest_events=1000, 
    **kwargs
):
    """
    Detects extremes using either a fixed or detrended baseline. 
    
    Use the Hobday et al. (2016, 2018) approach to detect extreme events
    with a relative threshold based on percentile. We use the 95th percentile
    as the default for the 1deg data. 

    Parameters
    ----------
    da: xr.DataArray
        A DataArray of values to detect extreme events 
    baseline_type: str [fixed/detrend]
        A string indicating the type of baseline. 
        If fixed, kwargs can include period=slice(start_year, end_year)
        If detrend, kwargs can include order=int indicating polynomial 
        order for detrending.
    quantile: float [0.95]
        A float between 0 and 1 to indicate the the extreme threshold
    baseline_period: slice(t0, t1) 
        A slice object indicating the period of the baseline for fixed baselines
    verbose: bool [True]
        prints out progress
        
    Returns
    -------
    xr.Dataset:
        A dataset that contains data, intensity, magnitude, and 
        normalised_intensity.

    See also
    --------
    lagrangian_event_filter
    """    
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    vprint('Creating baseline climatology and thresholds')
    if baseline_type == 'fixed':
        ds = fixed_baseline(da, quantile=quantile, **kwargs)
    elif baseline_type == 'detrend':
        ds = detrend_baseline(da, quantile=quantile, **kwargs)
    else:
        raise KeyError('`baseline_type` can only be: [fixed, detrend]')
    
    vprint('Doing some extreme statistics')
    magnitude = da - ds.climatology
    intensity = da - ds.threshold
    scaler = ds.threshold - ds.climatology
    normalised_intensity = magnitude / scaler
    
    ds['data'] = da
    ds['intensity'] = intensity.assign_attrs(description='peak over threshold')
    ds['magnitude'] = magnitude.assign_attrs(description='peak over mean')
    ds['intensity_norm'] = normalised_intensity.assign_attrs(
        description='(x - threshold) / (threshold - climatology)')

    mask = ds.intensity_norm > 1
    intensity = ds.intensity.where(mask)

    ds['blobs'] = simple_blob_detection(mask, n_largest=n_largest_events)
    ds['mask'] = ds.blobs.notnull() & mask
    
    ds = ds.assign_attrs(
        description=(
            'Extremes detected in the methods described in Hobday et al. '
            '(2016, 2018). If a shifting baseline is used, we detrend the '
            'data rather than using a true shifting baseline, as this '
            'allows for a longer baseline. Further, the full period is then '
            'used as the baseline. A fixed baseline uses a 30-year period. '
            'See global attributes for more details. '))
    
    return ds.astype('float32')


def simple_blob_detection(bool_mask, n_largest=1000):
    """
    Get the n largest blobs from a boolean mask and give them labels
    
    Uses the scipy.ndimage.label function to assign blob event labels. 
    
    Parameters
    ----------
    bool_mask: xr.DataArray(dtype=bool)
        A boolean mask that indicates where extremes are
    n_largest: int[1000]
        Choose only the n_largest events. Note that area per pixel is not 
        taken into account, only pixel count. 
        
    Returns
    -------
    xr.DataArray(dtype=float32)
        An array that contains labels of events. Non-events are masked as nans 
    """
    from scipy.ndimage import label
    import numpy as np
    import xarray as xr

    blobs, n_blobs = label(bool_mask)
    # returning the values and counts. Exclude the 1st value (not extremes)
    values, counts = np.array(np.unique(blobs, return_counts=True))[:, 1:]
    largest_n = values[counts.argsort()[-n_largest:]]
    mask = np.isin(blobs, largest_n)

    blobs = label(mask)[0]
    blobs = xr.DataArray(
        data=blobs,
        dims=bool_mask.dims, 
        coords=bool_mask.coords,
        attrs=dict(
            description=(
                'Blobs were created with scipy.ndimage.label with the '
                f'largest {n_largest} events being picked. No binary opening '
                'and closing is performed (as in the OceTrack package).'))
    ).where(mask)
    
    return blobs.astype('float32')