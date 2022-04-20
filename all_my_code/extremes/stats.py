try:
    from bottleneck import nanmean, nanpercentile, nansum
except ImportError:
    from numpy import nanmean, nanpercentile, nansum

import numpy as np


def quantile_95(x):
    """returns the 95th percentile of the data - does not work with nans"""
    return nanpercentile(x, 95)


def event_based_stats_2d_agg(
    da,
    dim="time",
    intra_extreme_func=nansum,
    inter_extreme_func=quantile_95,
    verbose=True,
    n_jobs=1,
):
    """
    Calculates event-based statistics and aggregates over the first
    dimension (presumably time).

    Can be used to calculate the 95th % of severity [default] or intensity.
    Extremes are defined as continuous values over the first dimension
    (separated by NaNs). For intensity, the intra_extreme_func should be
    mean/max/percentile.

    Parameters
    ----------
    da: xr.DataArray
        A masked array of intensity - if severity and intensity stats want
        to be calculated. The masked areas are non-extreme.
    intra_extreme_func: callable
        Applied within an extreme event. This function must operate over
        a single dimension and aggregate to a single value. It is not
        recommended to use a lambda function since the function name is
        not intelligible.
    inter_extreme_func: callable
        Applied between extreme events to aggregate over the first dimension
        in the dataarray. The function must operate over a single dimension
        and aggregate to a single value.
    n_jobs: int [1]
        the number of cpu's to use for the job (using Joblib). Note that
        a single thread can be nearly as fast as more than 1 job.
    verbose: int / bool
        the verbosity of Joblib's parallel function.

    Returns
    -------
    xr.DataArray:
        A 2D DataArray that has been aggregated over the first dimension
        (e.g. `time`). The output DataArray will have two new dimensions
        that represent the intra- and inter-extreme aggregating functions.

    Note
    ----
    This was tested on a small dataset (468, 180, 360) and might not work
    on larger datasets.
    """
    from joblib import Parallel, delayed
    from xarray import DataArray

    def event_aggregator(a, inter_event_agg_func=np.max, intra_event_agg_func=np.sum):
        """
        Calculates the event-based statistics over a single dimension.
        """

        def split_clumps_by_nan(a):
            masked = np.ma.masked_invalid(a)
            indicies = np.ma.clump_unmasked(masked)
            list_of_clumps = [a[s] for s in indicies]
            return list_of_clumps

        def aggregate_clumps(clumps, func=np.sum):
            clump_aggrates = np.array([func(clump) for clump in clumps])
            return clump_aggrates

        clumps = split_clumps_by_nan(a)
        severity = aggregate_clumps(clumps, func=intra_event_agg_func)
        aggregated_over_events = inter_event_agg_func(severity)
        return aggregated_over_events

    # first get the dimensions
    dims = list(da.dims)
    other_dims = dims.copy()
    other_dims.remove(dim)

    da = da.transpose(*([dim] + other_dims))

    print("finding nans")
    # we have to remove the nans
    mask = da.notnull().any("time").values.flatten()

    # the dataarray is unraveled on the first dim
    # transposed so that we iterate over the stacked dims
    print("reshaping")
    arr = da.values.reshape([da[dim].size, -1]).T
    assert mask.size == arr.shape[0]
    placeholder = np.ndarray(mask.size) * np.nan

    if n_jobs != 1:  # use Joblib if more than 1 job
        print("starting parallel job")
        func = delayed(event_aggregator)
        queue = [func(s, inter_extreme_func, intra_extreme_func) for s in arr[mask]]
        worker = Parallel(n_jobs=n_jobs, verbose=verbose)
        results = worker(queue)
    else:  # use a single thread if 1 job (often faster)
        print("starting single thread job")
        func = event_aggregator
        results = [func(s, inter_extreme_func, intra_extreme_func) for s in arr[mask]]
    # place the results in our placeholder that is not masked
    placeholder[mask] = np.array(results)

    # place the results in a data array and add coords/dims
    dims = dims[1:]
    results = DataArray(
        data=placeholder.reshape(*da.shape[1:]),
        coords={k: da.coords[k] for k in dims},
        dims=dims,
        attrs={
            "units": f'{da.attrs.get("units", "units")}/event',
            "description": (
                "events have been aggregated at two levels: intra- and inter event. "
                "Intra-event aggregation is the statistic used within an event. "
                "Inter-event aggregation is the statistic used between events. "
            ),
        },
    )
    # add coordinates that show the aggregation functions
    # we choose coordinates over attributes as these are
    # shown in xarray plots making it immediately clear what
    # the results show
    results = results.assign_coords(
        intra_event_func=intra_extreme_func.__name__,
        inter_event_func=inter_extreme_func.__name__,
    )

    return results


def duration(mask, aggregation_func=nanmean):
    """
    Calculates the duration of an event.
    """
    assert np.issubdtype(mask.dtype, np.bool), "Input must be a boolean array"

    da = mask.where(mask)

    duration = event_based_stats_2d_agg(
        da, intra_extreme_func=nansum, inter_extreme_func=aggregation_func
    ).rename("duration")

    duration = duration.assign_attrs(
        long_name="average_duration_of_extreme_events",
        units="months",
        description="average duration of extreme events",
    )
    return duration


def severity(intensity, aggregation_func=quantile_95):
    """
    Calculates the severity of an event.
    """
    da = intensity.where(intensity)
    severity = event_based_stats_2d_agg(
        da, intra_extreme_func=nansum, inter_extreme_func=quantile_95
    ).rename("severity")

    severity = severity.assign_attrs(
        long_name="95th_percentile_of_severity",
        description="The severity of an events as the sum of the intensity ",
    )

    return severity


def n_events(mask):
    """
    Calculates the number of events.
    """
    assert np.issubdtype(mask.dtype, np.bool), "Input must be a boolean array"

    da = mask.where(mask)

    n = event_based_stats_2d_agg(
        da, intra_extreme_func=nanmean, inter_extreme_func=nansum
    ).rename("n_events")

    n = n.assign_attrs(
        long_name="number_of_extreme_events",
        units="count",
        description="number of extreme events",
    )
    return n


# TODO: Add event-based (lagrangian) stats rather than eularian stats
