import xarray as xr
import numpy as np
from ..utils import apply_to_dataset


def _polyfit(da, **kwargs):
    """
    Calculate the polynomial fit of a time series without showing
    the RankWarning that often occurs when using polyfit
    """
    from warnings import catch_warnings, filterwarnings

    with catch_warnings():
        filterwarnings("ignore")
        return da.polyfit(**kwargs)


@apply_to_dataset
def slope(da, dim="time"):
    """
    Calculate the first order linear slope per {dim} unit

    Parameters
    ----------
    da: xr.DataArray
        the data to calculate the slope of
    dim: str [time]
        the dimension to calculate the slope over

    Returns
    -------
    xr.DataArray
        the slope of the data
    """
    # assign the dimension as step from 1 to the length of the dimension
    da = da.assign_coords(**{dim: np.arange(da[dim].size)})
    slope = (
        _polyfit(da, deg=1, dim=dim, skipna=True)
        .polyfit_coefficients[0]
        .drop("degree")
        .assign_attrs(units=f"units/{dim}_step")
    )
    slope = slope.append_attrs(
        history=f"Calculated linear slope along the `{dim}` dimension"
    )

    return slope


@apply_to_dataset
def climatology(da, tile=False, groupby_dim="time.month"):
    """
    Calculate the climatology of a time series

    Parameters
    ----------
    tile : bool
        If True, the output data will be the same size and have the
        same dimensions as the input data array
    groupby_dim : str
        Uses the xarray notation to group along a dimension.

    """

    dim0, dim1 = groupby_dim.split(".")
    clim = da.groupby(groupby_dim).mean(dim0)

    if tile:
        times = getattr(da[dim0].dt, dim1)
        clim = clim.sel(**{dim1: times}).drop(dim1)

    clim = clim.append_attrs(
        history=f"Calculated the `{dim1}` climatology along the `{dim0}` dimension"
    )

    return clim


@apply_to_dataset
def deseasonalise(da, groupby_dim="time.month", keep_mean=True):
    """
    Remove the seasonal cycle from the time series

    Parameters
    ----------
    da : xr.DataArray
        a data array for which you want to remove the seasonal cycle
    groupby_dim : str [time.month]
        the dimension to group by
    keep_mean : bool [True]
        if True, the mean of the data along the grouping dimension
        is kept in the output. If False, then the long term mean
        is removed.

    Returns
    -------
    da_deseasonalised : xr.DataArray
        the time series without the seasonal cycle
    """

    dim0, dim1 = groupby_dim.split(".")
    grp = da.groupby(groupby_dim)
    seasonal_cycle = grp.mean(dim0)
    if keep_mean:
        seasonal_cycle -= seasonal_cycle.mean(dim1)

    deseasonalised = grp - seasonal_cycle
    deseasonalised = deseasonalised.assign_attrs(units=da.attrs.get("units", ""))

    deseasonalised = deseasonalised.append_attrs(
        history=f"Removed the `{dim1}` climatology along the `{dim0}` dimension"
    )

    return deseasonalised


@apply_to_dataset
def trend(da, dim="time", deg=1, coef=None):
    """
    The trend over the given dimension

    Will mask nans where all nans along the dimension

    Parameters
    ----------
    da : xr.DataArray
        a data array for which you want to calculate the trend
    dim : str [time]
        the dimension along which the trend will be calculated
    deg : int [1]
        the order of the polynomial used to fit the data
    coef : xr.DataArray
        the coefficients of the polynomial fit if precalculated
        to save on computation time

    Returns
    -------
    trend : predicted trend data over the given period and dimensions
    """

    if coef is None:
        coef = _polyfit(da, deg=deg, dim=dim).polyfit_coefficients

    # use the coeficients to predict along the dimension
    trend = xr.polyval(da[dim], coef)

    # mask nans where all are nan along a given dimension
    mask = da.notnull().any(dim)
    trend = trend.where(mask)

    name = getattr(da, "name", None)
    name = name + "_" if name is not None else ""
    trend.name = name + "predicted_trend"

    trend = trend.append_attrs(
        history=f"Calculated the {deg}-order trend along the `{dim}` dimension"
    )

    return trend


@apply_to_dataset
def detrend(da, dim="time", deg=1, coef=None):
    """
    Remove the trend along the [time] dimension

    Parameters
    ----------
    dim : str [time]
        the dimension along which the trend will be calculated
    deg : int [1]
        the order of the polynomial used to fit the data
    coef : xr.DataArray
        the coefficients of the polynomial fit if precalculated
        to save on computation time

    Returns
    -------
    detrended : the trends of the input at the given order
    """

    da_detrend = da - trend(da, dim=dim, deg=deg, coef=coef)
    da_detrend = da_detrend.append_attrs(
        history=f"Removed the {deg}-order trend along the `{dim}` dimension"
    )

    return da_detrend


@apply_to_dataset
def corr(da1, da2, lag=0, dim="time"):
    """
    Calculate the correlation between two xarray.DataArray objects.

    Parameters
    ----------
    da1: xr.DataArray
        The first data array
    da2: xr.DataArray
        The second data array
    lag: int or list of ints [0]
        The lag between the two data arrays
    dim: str
        The name of the time dimension

    Returns
    -------
    xr.DataArray
        The correlation between the two data arrays
    """
    from pandas.api.types import is_list_like

    assert isinstance(da2, xr.DataArray), "da2 must be an xarray.DataArray"

    if is_list_like(lag):
        assert all(
            isinstance(x, (np.int_, int)) for x in lag
        ), "all entries in lag must be integers"
        lagged = xr.concat([da2.shift(**{dim: x}) for x in lag], "lag").assign_coords(
            lag=lag
        )
        return xr.corr(da1, lagged, dim=dim)
    elif isinstance(lag, (np.int_, int)):
        correlated = corr(da1, da2, lag=[lag], dim=dim)
        return correlated.sel(lag=lag, drop=True)


@apply_to_dataset
def auto_corr(da, lag, dim="time"):
    """
    Calculate the autocorrelation of a xarray.DataArray object.

    Parameters
    ----------
    da: xr.DataArray
        The data array
    lag: int
        The lag applied to the data array. Will range from negative
        to positive of this value
    dim: str
        The name of the time dimension

    Returns
    -------
    xr.DataArray
        The autocorrelation of the data array
    """
    assert isinstance(lag, int), "lag must be an integer"
    lags = np.arange(-lag, lag + 1, 1).astype(int)
    correlated = corr(da, da, lag=lags, dim=dim)
    return correlated


def polyfit(
    y, x=None, dim="time", deg=1, full=True, drop_polyfit_name=True, skipna=True
):
    """
    Full linear regression with all stats (coefs, r2, pvalue, rmse, +)

    Only slightly slower than using xr.DataArray.polyfit, but returns
    all stats (if full=True)

    Parameters
    ----------
    y : xr.DataArray
        the dependent variable
    x : xr.DataArray
        if None, then dim will be used, otherwise x will be used
    dim : str [time]
        the dimension along which the trend will be calculated
    deg : int [1]
        the order of the polynomial used to fit the data
    full : bool [True]
        if True, the full regression results will be returned

    Returns
    -------
    polyfit : xr.DataArray
        the linear regression results containing coefficients,
        rsquared, pvalue, and rmse
    """
    from scipy.stats import beta
    from xarray import DataArray, Dataset

    if isinstance(y, Dataset):
        from ..utils import run_parallel

        inputs = [y[k].rename(k) for k in y]
        outputs = run_parallel(
            polyfit,
            inputs,
            kwargs=dict(
                dim=dim, deg=deg, full=full, drop_polyfit_name=False, skipna=skipna
            ),
            verbose=True,
        )

        return xr.merge(outputs)

    # total sum of squares (use this for non-agg dimensions)
    tss = np.square(y - y.mean(dim)).sum(dim)

    if x is not None:
        if isinstance(x, DataArray):
            xx = x.conform.drop_0d_coords().dropna(dim, how="all")
            xname = x.name
        else:
            xx = DataArray(data=x, dims=[dim], coords={dim: y[dim]})
            xname = "x"

        # create regession array
        coords = {k: tss[k].values for k in tss.coords}
        coords[xname] = xx.values
        yy = xr.DataArray(
            data=y.sel(**{dim: xx[dim]}).values,
            dims=[xname] + list(tss.dims),
            coords=coords,
        )
        dim = xname
    else:
        assert dim in y.dims, "given dimension is not in y"
        yy = y

    # calculate polyfit
    fit = _polyfit(yy, dim=dim, deg=deg, full=full, skipna=skipna)

    if not full:
        return fit

    # residual sum of squares
    rss = fit.polyfit_residuals

    fit["polyfit_rsquared"] = (1 - (rss / tss)).assign_attrs(
        description="pearsons r-value squared"
    )
    r = fit["polyfit_rsquared"] ** 0.5

    n = yy[dim].size
    # creating the distribution for pvalue
    dist = beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)

    # calculating R value
    fit["polyfit_pvalue"] = (r * np.nan).fillna(2 * dist.cdf(-abs(r)))
    fit["polyfit_rmse"] = (rss / n) ** 0.5

    name = getattr(y, "name", None)
    if drop_polyfit_name:
        rename_dict = {k: k.replace("polyfit_", "") for k in fit}
        fit = fit.rename(rename_dict)
    elif name is not None:
        rename_dict = {k: f"{name}_{k}" for k in fit}
        fit = fit.rename(rename_dict)

    fit = fit.append_attrs(
        history="Full linear regression with all stats (coefs, r2, pvalue, rmse, +)"
    )

    return fit


@apply_to_dataset
def linregress(y, x=None, dim="time", skipna=True):
    """
    Calculate the linear regression between two xarray.DataArray objects.

    Parameters
    ----------
    y : xarray.DataArray
        The dependent variable.
    x : xarray.DataArray, optional
        The independent variable. If not provided, the function will
        calculate the linear regression between the time dimension of
        y and the time dimension of y.
    dim : str, optional
        The dimension along which to calculate the linear regression.

    Returns
    -------
    xarray.DataArray
        The linear regression coefficient, intercept, and r-value.
    """
    from xskillscore import pearson_r, linslope, pearson_r_p_value, rmse, r2, mape

    if isinstance(y, xr.Dataset):
        raise TypeError("linregress is not implemented for xr.Dataset")

    if skipna:
        grid = y.isel(**{dim: 0}).copy()
        mask = y.notnull().any(dim)
        y = y.where(mask, drop=True)

    if x is None:
        x = xr.DataArray(np.arange(y[dim].size), dims=dim, coords={dim: y[dim]})
        x = xr.ones_like(y) * x

    ds = xr.Dataset()
    ds["x"] = x
    ds["y"] = y
    ds["rvalue"] = pearson_r(x, y, dim=dim, skipna=skipna)
    ds["slope"] = linslope(x, y, dim=dim, skipna=skipna)
    ds["intercept"] = y.mean(dim=dim) - ds.slope * x.mean(dim=dim)
    ds["yhat"] = x * ds.slope + ds.intercept
    # the following all need yhat as input
    ds["r2"] = r2(y, ds.yhat, dim=dim, skipna=skipna)
    ds["rmse"] = rmse(y, ds.yhat, dim=dim, skipna=skipna)
    ds["pvalue"] = pearson_r_p_value(y, ds.yhat, dim=dim, skipna=skipna)
    ds["mape"] = mape(y, ds.yhat, dim=dim, skipna=True) * 100

    if skipna:
        ds = ds.reindex_like(grid)
    order = "x y yhat slope intercept rvalue r2 pvalue rmse mape".split()
    return ds[order]


@apply_to_dataset
def time_of_emergence_stdev(
    da, deseasonalise=True, noise_multiplier=2, detrend_poly_order=1, dim="time"
):
    """
    Calculate time of emergence based on standard deviation

    Parameters
    ----------
    da : xr.DataArray
        data for which to calculate the ToE
    type : str [deseason | seasonal_cycle]
        determines if the "noise" of the ToE calculation is based on
        the standard deviation of the detrended data or based on the
        standard deviation of the seasonal cycle
    noise_multiplier : float [2]
        how much confidence you want - sigma * 2 = 95th percentile
    dim : str [time]
        dimension along which ToE is calculated

    Returns
    -------
    time_of_emergence : xr.DataArray
        the time of emergence in years

    Reference
    ---------
    Keller, K. M., F. Joos, and C. C. Raible (2014),
        Time of emergence of trends in ocean biogeochemistry,
        Biogeosciences, 11(13), 3647–3659,
        doi:10.5194/bg-11-3647-2014.
    """

    name = getattr(da, "name", None)
    name = name + "_" if name is not None else ""

    noise_in = da.time_series.detrend(deg=detrend_poly_order, dim=dim)
    if deseasonalise:
        noise_in = noise_in.time_series.deseasonalise()

    noise = noise_in.std(dim) * noise_multiplier
    slope = da.resample(**{dim: "1AS"}).mean().time_series.slope(dim=dim)

    toe = (noise / abs(slope)).assign_attrs(
        description=("time of emergence in years (based on standard deviation) "),
        long_name=f"{name}time_of_emergence",
        units="years",
    )
    toe = toe.append_attrs(
        history="Time of Emergence calculated using standard deviation"
    )

    return toe


@apply_to_dataset
def decompose_modes_of_variability(
    da,
    time_dim="time",
    seasonal_dim="month",
    interannual_smoother_window_size=None,
    detrend=False,
):
    """
    Split a time-series into different modes of variability:
    subseasonal, seasonal, and interannual

    Parameters
    ----------
    da: xr.DataArray
        A data array with a time index/axes
    time_dim: str
        the name of the time dimension (must be a np.datetime64 dim)
    seasonal_dim: str
        the name of the time step over which to group for
        the seasonal cycle variability
    interannual_smoother_window_size: None | int
        the window size of the smoother that is applied to the
        deseasonalised data. If None is given, then it will be
        a one year smoother
    detrend: bool
        will detrend with linear regression the data if True,
        otherwise, the trend will be included

    Returns
    -------
    xr.Dataset | xr.DataArray (depending on input type):
        The output will have an additional dimension called `mode_of_var`
        with length 4 that contains the following: original_input,
        subseasonal, seasonal, interannual.
    """
    if isinstance(da, xr.DataArray):
        seas_time = getattr(da[time_dim].dt, seasonal_dim)

        if detrend:
            da = da.stats.detrend()
            prefix = "detrended"
        else:
            prefix = "original"
        grp = da.groupby(f"{time_dim}.{seasonal_dim}")

        if interannual_smoother_window_size is None:
            t = len(grp.groups) * 1
        elif isinstance(interannual_smoother_window_size, int):
            t = interannual_smoother_window_size
        else:
            raise TypeError("`interannual_smoother_window_size` must be None or int")

        ds = xr.Dataset()
        seasonal_cycle = grp.mean()
        deseasonalised = grp - seasonal_cycle
        interannual = deseasonalised.rolling(
            **{time_dim: t}, center=True, min_periods=int(t / 4)
        ).mean()

        ds[f"{prefix}_input"] = da
        ds["subsesasonal"] = deseasonalised - interannual
        ds["seasonal"] = seasonal_cycle.sel(**{seasonal_dim: seas_time}).drop(
            seasonal_dim
        )
        ds["interannual"] = interannual

        ds = ds.to_array(dim="mode_of_var")
        return ds


def interannual_variability(da, dim="time"):
    """
    Calculate the interannual variability of a time series

    Parameters
    ----------
    da : xr.DataArray
        a data array for which you want to calculate the interannual variability

    Returns
    -------
    interannual_variability : xr.DataArray
        the interannual variability of the input
    """

    # calculate the mean of the data
    da_annual = da.resample(time="1AS", loffset="182D").mean()
    da_detrend = da_annual.time_series.detrend()

    # calculate the standard deviation of the data
    interannual_variability = da_detrend.std(dim)

    return interannual_variability


def anom(da, dim="time", ref=None):
    """
    Calculate the anomaly of a time series

    Parameters
    ----------
    da : xr.DataArray
        a data array for which you want to calculate the anomaly
    ref : int | None
        the reference year for the anomaly calculation
        if None, the the mean of the period is chosen

    Returns
    -------
    anom : xr.DataArray
        the anomaly of the input
    """

    if ref is None:
        ref = da.mean(dim=dim)
    elif isinstance(ref, int):
        ref = da.isel(**{dim: ref})
    else:
        raise ValueError("ref must be an int or None")

    anom = da - ref

    return anom
