import numpy as np
import xarray as xr


def fit_seasonal_cycle_harmonic(da, target_time=12, dim="time", window_years=3):
    """fits a seasonal cycle harmonic to data

    Parameters
    ----------
    da : xr.DataArray
        a 1D dataset with time as the dimension
    dim : str
        time dimension
    target_time : pd.DatetimeIndex, float
        can be either a float or a pd.DatetimeIndex. if a float,
        it represents the size of the time step as a fraction of a
        year. If it's a pd.DatetimeIndex, then the new output will
        be interpolated onto that new time stamp.

    Returns
    -------
    xr.DataArray
        A fitted output with time either as target_time as inputted,
        or if target_time is an integer, then as a datetime array
        centered roughly to the window width of the time step.
    """
    from scipy.optimize import curve_fit
    from numpy import cos, sin, pi

    # from numba import njit
    from pandas import Series, DatetimeIndex

    # @njit
    def fit_seascycle_harmonic(x, a1, a2, a3, a4, a5, a6, a7):
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

    # decimal year
    decimal_year = da.time.dt.dayofyear / 366 + da.time.dt.year
    da_na = da.assign_coords(time=decimal_year).dropna("time")

    y = da_na.data
    x = da_na.time.data

    t0 = x.min() // 1
    t1 = x.max() // 1

    if isinstance(target_time, DatetimeIndex):
        time_out = target_time.dayofyear / 366 + target_time.year
    elif isinstance(target_time, (int, float)):
        time_step = 1 / target_time
        time_out = np.arange(t0 + time_step / 2, t1, time_step)
    else:
        raise TypeError("target_time must be int or pd.DateTimeIndex")

    fitted = {}
    for i0 in np.arange(t0, t1 + 1):
        i1 = i0 + window_years

        # mask limits x and y to window years
        wmask = (x > i0) & (x < i1)
        coefs = curve_fit(
            fit_seascycle_harmonic,
            x[wmask],
            y[wmask],
            p0=[300, 1.1, 0.01, -3, -7, 5.5, 5.5],
        )[0]

        # create estimates
        new_x = time_out[(time_out >= i0) & (time_out <= i1)].astype(float)
        new_y = fit_seascycle_harmonic(new_x, *coefs)

        new = {_decimal_year_to_datetime(k): v for k, v in zip(new_x, new_y)}

        # now the new data needs to be added to the specific time (k)
        for k in new:
            if k not in fitted:
                fitted[k] = [new[k]]
            else:
                fitted[k] += (new[k],)
    # for each key (time), take the average of the 3 or less points

    for k in fitted:
        fitted[k] = np.nanmean(fitted[k])
    # convert the fittedput to an xarray.dataarray
    fitted = Series(fitted).to_xarray().rename(index="time")

    return fitted


def seascycl_fit_graven(da, n_years=3, dim="time"):
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
        The data to fit a seasonal cycle to. Time dimension must be a time index
    n_years : int
        The number of years to fit the seasonal cycle to (a rolling window)
    dim : str
        The name of the time dimension. Must be a time index

    Returns
    -------
    xarray.Dataset
        The fitted seasonal cycle and the difference between the JJA and DJF
    """

    def get_months_from_time(time, months, tile=1):
        """
        gets the index of the given months in the time array
        will only give index for first year unless tile > 1
        """

        # get the unique years from the time array with counts
        years = time.dt.year.values
        unique, counts = np.unique(years, return_counts=True)

        # assert that all counts are the same
        msg = "this function does not work for unevenly spaced time"
        assert np.all(counts == counts[0]), msg
        n_steps = counts[0]

        year_month = time.dt.month.values[years == unique[0]]

        # get the months that are in the list
        bool_idx = np.isin(year_month, months)
        # get the indices of the months that are in the list
        loc_idx = np.where(bool_idx)[0]

        # tile the indicies so the fit the given tile
        idxs = [loc_idx + i * n_steps for i in range(tile)]
        idxs = np.concatenate(idxs)

        return idxs

    from numba import njit
    from numpy import sin, cos, pi

    stride = _get_number_of_time_steps_in_year(da[dim])
    window = n_years * stride

    assert stride == 12, "this function only works for monthly data"
    assert n_years % 2, "n_years must be an odd number"

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
        # we do not center since this shifts the months by 6
        da.rolling(**{dim: window}, center=False, min_periods=stride)
        .construct(**{dim: "time_step"}, stride=stride)
        .stack(other=dims)
        .where(lambda x: x.notnull().sum("time_step") > stride, drop=True)
        .assign_coords(time_step=(np.arange(window) % stride + 1) / stride)
    )

    fast_func = njit()(fit_sc)
    coefs = windowed.curvefit(
        coords="time_step",
        func=fast_func,
        p0=[300, 1.1, 0.01, -3, -7, 5.5, 5.5],
        kwargs={"maxfev": 100},
    )

    seas_cycle = (
        # multiply out coefficients
        fit_sc(windowed.time_step, *coefs.curvefit_coefficients.T)
        .drop("param")
        .assign_coords(time_step=lambda x: x.time_step * stride)
        .groupby("time_step")
        .mean()
        .unstack()
    )

    idx_jja = get_months_from_time(da.time, [6, 7, 8])
    idx_djf = get_months_from_time(da.time, [12, 1, 2])
    jja = seas_cycle.isel(time_step=idx_jja).mean(dim="time_step")
    djf = seas_cycle.isel(time_step=idx_djf).mean(dim="time_step")

    out = xr.Dataset()
    out["seas_cycle"] = seas_cycle
    out["jja_minus_djf"] = jja - djf

    return out


def seascycl_fit_climatology(da, n_years=3, dim="time"):
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

    stride = _get_number_of_time_steps_in_year(da[dim])
    assert stride == 12, "this function only works for monthly data"

    window = n_years * stride

    dims = list(da.dims)
    dims.remove(dim)
    seas_cycle = (
        # we do not center since this shifts the months by 6
        da.rolling(**{dim: window}, center=False, min_periods=stride)
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


def _get_number_of_time_steps_in_year(time, raise_if_uneven=True):
    """
    Get the number of time steps in a year (e.g. months, days, etc.)
    """
    # get the unique years from the time array with counts
    years = time.dt.year.values
    unique, counts = np.unique(years, return_counts=True)

    all_the_same = np.all(counts == counts[0])
    if not all_the_same and raise_if_uneven:
        raise ValueError(f"time array is not evenly spaced: {time}")
    else:
        return counts[0]


def _decimal_year_to_datetime(decimal_year):
    from datetime import datetime, timedelta
    from pandas import Timestamp

    t = decimal_year
    year = int(t)
    ndays_in_year = 365 if (year % 4) else 366
    day = int(np.around((t - year) * ndays_in_year)) - 1
    time = Timestamp(datetime(int(year), 1, 1) + timedelta(days=day))

    return time
