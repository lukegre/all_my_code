import xarray as xr


def fixed_baseline(
    da,
    quantile=0.95,
    period=slice("1985", "2014"),
    clim_agg_func="mean",
    n_largest_events=1000,
):
    """
    Detects extreme events using a fixed baseline.

    Parameters
    ----------
    da: xr.DataArray
        A DataArray of values to detect extreme events
    quantile: float [0.95]
        The quantile to use for the threshold
    period: slice [1985:2014]
        The period to use for the baseline
    clim_agg_func: str ['mean']
        The function to use to aggregate the climatology - Hobday uses mean,
        but median could make more sense
    n_largest_events: int [1000]
        The number of events to use for the blob detection.

    Returns
    -------
    xr.Dataset:
        A dataset that contains data, intensity, magnitude,
        intensity_norm, mask, and blobs. Details of each variable
        are also given in variable descriptions
        intensity = peak over threshold
        magnitude = peak over baseline
        intensity_norm = intensity / (threshold / baseline)
        mask = True where (intensity_norm > 1) & (num blobs == 1000)
        blobs = blob intiger labels for the mask

    See also
    --------
    poly_baseline
    """
    baseline = da.sel(time=period)

    grp = baseline.groupby("time.month")
    thresh = grp.quantile(quantile, "time").sel(month=da.time.dt.month)
    clim = getattr(grp, clim_agg_func)("time").sel(month=da.time.dt.month)
    attrs = dict(
        baseline_type="fixed",
        baseline_period=f"{period.start}:{period.stop}",
        threshold_quantile=quantile,
    )

    ds = xr.Dataset()
    ds["data"] = da
    ds["threshold"] = thresh.assign_attrs(attrs)
    ds["baseline"] = clim.assign_attrs(aggregation_function=clim_agg_func, **attrs)

    ds = _add_derived_vars(ds, n_largest_events=n_largest_events)
    ds = ds.assign_attrs(attrs)

    return ds


def poly_baseline(
    da, deg=1, quantile=0.95, clim_agg_func="mean", n_largest_events=1000
):
    """
    Detects extreme events using a baseline that is based on a polynomial.

    Parameters
    ----------
    da: xr.DataArray
        A DataArray of values to detect extreme events
    deg: int [1]
        The degree/order of the polynomial to use for the baseline
    quantile: float [0.95]
        The quantile to use for the threshold
    clim_agg_func: str ['mean']
        The function to use to aggregate the climatology - Hobday uses mean,
        but median could make more sense
    n_largest_events: int [1000]
        The number of events to use for the blob detection.

    Returns
    -------
    xr.Dataset:
        A dataset that contains data, intensity, magnitude,
        intensity_norm, mask, and blobs. Details of each variable
        are also given in variable descriptions
        intensity = peak over threshold
        magnitude = peak over baseline
        intensity_norm = intensity / (threshold / baseline)
        mask = True where (intensity_norm > 1) & (num blobs == 1000)
        blobs = blob intiger labels for the mask

    See also
    --------
    fixed_baseline
    """

    assert "time" in da.coords, "time must be in `da` as a coordinate and dimension"

    trend = da.time_series.trend(deg=deg)
    baseline = da - trend
    grp = baseline.groupby("time.month")
    thresh = grp.quantile(quantile, "time").sel(month=da.time.dt.month)
    clim = getattr(grp, clim_agg_func)("time").sel(month=da.time.dt.month)

    attrs = dict(
        baseline_type="polynomial",
        baseline_poly_deg=deg,
        baseline_period=f"{da.time.dt.year.values[0]}:{da.time.dt.year.values[-1]}",
        threshold_quantile=quantile,
    )

    ds = xr.Dataset()
    ds["data"] = da
    ds["threshold"] = (thresh + trend).assign_attrs(**attrs)
    ds["baseline"] = (clim + trend).assign_attrs(func=clim_agg_func, **attrs)

    ds = _add_derived_vars(ds, n_largest_events=n_largest_events)
    ds = ds.assign_attrs(attrs)

    return ds


def _add_derived_vars(ds, n_largest_events=1000):
    """
    Adds derived varaibles to the existing extreme dataset
    The same for all dtection approaches

    Parameters
    ----------
    da: xr.DataArray
        A DataArray of values to detect extreme events
    n_largest_events: int [1000]
        The number of events to use for the blob detection.

    Returns
    -------
    xr.Dataset:
        A dataset that contains data, intensity, magnitude, and
        intensity_norm, mask, and blobs. Details of each variable
        are also given in variable descriptions
        intensity = peak over threshold
        magnitude = peak over baseline
        intensity_norm = intensity / (threshold / baseline)

    See also
    --------
    _simple_blob_detection
    """

    magnitude = ds.data - ds.baseline
    intensity = ds.data - ds.threshold
    scaler = ds.threshold - ds.baseline
    normalised_intensity = magnitude / scaler

    ds["intensity"] = intensity.assign_attrs(description="peak over threshold")
    ds["magnitude"] = magnitude.assign_attrs(description="peak over mean")
    ds["intensity_norm"] = normalised_intensity.assign_attrs(
        description="(x - threshold) / (threshold - baseline)"
    )

    tmp_mask = ds.intensity_norm > 1
    intensity = ds.intensity.where(tmp_mask)
    ds = ds.astype("float32")

    ds["blobs"] = _simple_blob_detection(tmp_mask, n_largest=n_largest_events).astype(
        int
    )
    ds["mask"] = ds.blobs.notnull() & tmp_mask

    ds = ds.assign_attrs(
        description=(
            "Extremes detected in the methods described in Hobday et al. "
            "(2016, 2018). If a shifting baseline is used, we detrend the "
            "data rather than using a true shifting baseline, as this "
            "allows for a longer baseline. Further, the full period is then "
            "used as the baseline. A fixed baseline uses a 30-year period. "
            "See global attributes for more details. "
        )
    )

    ds = ds.drop("month", errors="ignore")
    return ds


def _simple_blob_detection(bool_mask, n_largest=1000):
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
                "Blobs were created with scipy.ndimage.label with the "
                f"largest {n_largest} events being picked. No binary opening "
                "and closing is performed (as in the OceTrack package)."
            )
        ),
    ).where(mask)

    return blobs.astype("float32")
