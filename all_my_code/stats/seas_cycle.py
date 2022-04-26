import numpy as np
import xarray as xr


def seascycl_fit_graven(da, window=36, stride=12, dim="time"):
    """
    Fits a seasonal cycle to data using cos and sin functions.

    Using the approach defined in Graven et al. (2013)

    Note
    ----
    This function is slow with large datasets - it is recommended to use
    `seascycl_fit_climatology` when the dataset is large. This function
    is suited to small datasets with sparse data.

    Parameters
    ----------
    da : xarray.DataArray
        The data to fit a seasonal cycle to
    window : int
        The number of months to fit the seasonal cycle to
    stride : int
        The number of months to advance the window.
    dim : str
        The dimension to use for the window.

    Returns
    -------
    xarray.Dataset
        The fitted seasonal cycle and the difference between the JJA and DJF
    """
    from numba import njit
    from numpy import sin, cos, pi

    assert window % stride == 0, "window must be a multiple of stride"
    assert (window / stride) % 2, "window / stride must be an odd number"

    def fit_sc(x, a1, a2, a3, a4, a5, a6, a7):
        """function to fit as defined by Peter"""
        return (
            a1
            + a2 * x
            + a3 * x**2
            + a4 * sin(2 * pi * x)
            + a5 * cos(2 * pi * x)
            + a6 * sin(4 * pi * x)
            + a7 * cos(4 * pi * x)
        )

    dims = list(da.dims)
    dims.remove(dim)
    windowed = (
        da.rolling(**{dim: window}, center=True, min_periods=stride)
        .construct(**{dim: "month"}, stride=stride)
        .stack(other=dims)
        .where(lambda x: x.notnull().sum("month") > 12, drop=True)
        .assign_coords(month=np.arange(window) % 12 + 1)
    )

    fast_func = njit()(fit_sc)
    coefs = windowed.curvefit(
        coords="month",
        func=fast_func,
        p0=[300, 1.1, 0.01, -3, -7, 5.5, 5.5],
        kwargs={"maxfev": 100},
    )

    seas_cycle = (
        fit_sc(windowed.month, *coefs.curvefit_coefficients.T)
        .drop("param")
        .assign_coords(month=lambda x: x.month)
        .groupby("month")
        .mean()
        .unstack()
    )

    mon_avg = lambda x, m: x.sel(month=m).mean("month")
    out = xr.Dataset()
    out["seas_cycle"] = seas_cycle
    out["jja_minus_djf"] = mon_avg(seas_cycle, [6, 7, 8]) - mon_avg(
        seas_cycle, [12, 1, 2]
    )

    return out


def seascycl_fit_climatology(da, window=36, stride=12, dim="time"):
    """
    Fit a seasonal cycle to the climatology of a time series.

    Parameters
    ----------
    da : xarray.DataArray
    window : int
        The number of months in the window.
    stride : int
        The number of months to advance the window.
    dim : str
        The dimension to use for the window.

    Returns
    -------
    xarray.Dataset
        The seasonal cycle and the difference between the JJA and DJF
    """
    assert window % stride == 0, "window must be a multiple of stride"
    assert (window / stride) % 2, "window / stride must be an odd number"

    dims = list(da.dims)
    dims.remove(dim)
    seas_cycle = (
        da.rolling(**{dim: window}, center=True, min_periods=stride)
        .construct(**{dim: "month"}, stride=stride)
        .assign_coords(month=np.arange(window) % 12 + 1)
        .groupby("month")
        .mean()
    )

    mon_avg = lambda x, m: x.sel(month=m).mean("month")
    out = xr.Dataset()
    out["seas_cycle"] = seas_cycle
    out["jja_minus_djf"] = mon_avg(seas_cycle, [6, 7, 8]) - mon_avg(
        seas_cycle, [12, 1, 2]
    )

    return out
