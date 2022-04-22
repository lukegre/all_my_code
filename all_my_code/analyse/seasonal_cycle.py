import xarray as xr
import numpy as np
import pandas as pd


def _fit_graven2013_monthly_numpy(y):
    """
    Fits a seasonal cycle to data using cos and sin functions.
    Accounts for annual and 6-monthly cycles.

    Args:
        y (array-like): the time-varying value to fit a seasonal cycle to

    Returns:
        array-like: a fitted seasonal cycle (y-hat)
    """
    from scipy import optimize
    from numpy import sin, cos, pi, nan, isnan, ndarray

    if isnan(y).any():
        return ndarray([12]) * nan

    x = np.arange(y.size) / 12

    def func(t, a1, a2, a3, a4, a5, a6, a7):
        """function to fit as defined by Peter"""
        return (
            a1
            + a2 * t
            + a3 * t**2
            + a4 * sin(2 * pi * t)
            + a5 * cos(2 * pi * t)
            + a6 * sin(4 * pi * t)
            + a7 * cos(4 * pi * t)
        )

    p0 = 300, 1.1, 0.01, -3, -7, 5.5, 5.5
    try:
        params, params_covariance = optimize.curve_fit(func, x, y, p0=p0)
    except RuntimeError:
        params = [nan] * 7
    yhat = func(x, *params)
    yhat = yhat.reshape(int(y.size / 12), 12).mean(axis=0)

    return yhat


def fit_seasonal_cycle_graven2013(da):
    def wrap_graven(y):
        d0 = y.shape[-1]
        dn = y.shape[:-1]
        out = map(_fit_graven2013_monthly_numpy, y.reshape(-1, d0))
        out = np.array(list(out)).reshape(*dn, -1)

        return out

    dims = list(da.dims)
    dims.remove("time")
    out = xr.apply_ufunc(
        wrap_graven,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["month"]],
        dask_gufunc_kwargs=dict(output_sizes={"month": 12}),
        dask="parallelized",
        output_dtypes=[float],
    )

    out = out.transpose("month", *dims).assign_coords(month=np.arange(1, 13))

    return out


def fit_seasonal_cycle_wnt_smr(da):
    return da.groupby("time.month").mean()


def fit_rolling_seasonal_cycle(da, window=36, func=fit_seasonal_cycle_wnt_smr):
    """
    Fits a seasonal cycle to data using cos and sin functions.

    Parameters
    ----------
    da : xarray.DataArray
        The data to fit a seasonal cycle to
    window : int
        The number of months to fit the seasonal cycle to
    func : function
        The function to fit the seasonal cycle to

    Returns
    -------
    xarray.Dataset
        The fitted seasonal cycle
    """

    assert isinstance(da, xr.DataArray), "da must be an xarray DataArray"
    assert window % 12 == 0, "window must be a multiple of 12 (months)"
    assert "time" in da.dims, "da must have a time dimension"
    assert "lon" in da.dims, "da must have a lon dimension"
    assert "lat" in da.dims, "da must have a lat dimension"

    da = da.conform.time_center_monthly(center_day=1)
    w = window
    years = w / 12
    freq = f"{int(years)}AS"
    offset = int(years / 2)

    inputs = [
        da.isel(time=slice(i, None))
        .resample(time=freq, label="left", loffset=f"{offset}AS")
        .apply(func)
        for i in range(0, w, 12)
    ]

    dims = list(da.dims)
    dims.insert(1, "month")
    out = xr.concat(inputs, "time").sortby("time").transpose(*dims)

    mon, year = [a.flatten() for a in np.meshgrid(out.month, out.time.dt.year)]
    times = pd.to_datetime([f"{y}-{m:02d}" for y, m in zip(year, mon)])

    coords = dict(da.coords)
    coords.update(time=times)
    dims = list(da.dims)
    dims.remove("time")

    seasonal_fit = xr.DataArray(
        data=out.values.reshape(*da.shape), coords=coords, dims=da.dims
    ).where(da.notnull().any(dims))

    seasonal = seasonal_fit.resample(time="1Q-FEB").mean()
    djf = (
        seasonal[0::4]
        .assign_coords(time=lambda x: x.time.dt.year.values)
        .rename(time="year")
    )
    jja = (
        seasonal[2::4]
        .assign_coords(time=lambda x: x.time.dt.year.values)
        .rename(time="year")
    )

    ds = xr.Dataset()
    ds["seasonal_clim_fit"] = seasonal_fit
    ds["rmse"] = ((da - seasonal_fit) ** 2).mean("time") ** 0.5
    ds["seas_jja_minus_djf"] = jja - djf

    return ds
