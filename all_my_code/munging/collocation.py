import numpy as np
import pandas as pd
import xarray as xr
from .. import logger


def grid_flat_data(*data_columns, sparse=True, **coordinate_columns):
    """
    Takes data columns and grids them, based on semi-discrete (binned)
    coordinate columns.

    Parameters
    ----------
    data_columns : array-like
        columns that will be gridded. Must have the same length as
        the coordinate columns
    return_dataarray : bool [True]
        if True, returns a dataarray as xarray.DataArray or xarray.Dataset
    """

    sizes = np.unique([len(v) for v in coordinate_columns.values()])
    assert len(sizes) == 1, "All coordinates need to be the same length"

    coordinate_cols = {k: v.values for k, v in coordinate_columns.items()}
    coordinate_cols.update({v.name: v for v in data_columns})

    df = pd.DataFrame.from_dict(coordinate_cols)
    grouped = df.groupby(by=coordinate_cols)
    out = grouped.mean()

    if isinstance(out, xr.DataArray):
        return xr.DataArray.from_series(out, sparse=sparse)
    elif isinstance(out, xr.Dataset):
        return xr.Dataset.from_dataframe(out, sparse=sparse)

    return out


def _grid_flat_data(*data_columns, return_dataaray=True, **coordinate_columns):
    """
    Takes data columns and grids them, based on semi-discrete (binned)
    coordinate columns.

    Parameters
    ----------
    data_columns : array-like
        columns that will be gridded. Must have the same length as
        the coordinate columns
    return_dataarray : bool [True]
        if True, returns a dataarray as xarray.DataArray or xarray.Dataset
    """

    def make_bins(a):
        if isinstance(a[0], np.datetime64):
            istime = True
            inf = np.timedelta64(9, "Y").astype("timedelta64[ns]").astype(float)
        else:
            istime = False
            inf = np.inf

        a = a.astype(float) if istime else a

        a = np.convolve(a, [0.5] * 2, "valid")
        a = np.r_[a[0] - inf, a, a[-1] + inf]

        if istime:
            a = a.astype(float).astype("datetime64[ns]")
        return a

    def get_index_labels(a):
        uniq = np.unique(a)
        bins = make_bins(uniq)
        labels = np.arange(uniq.size)
        ind = pd.cut(pd.Series(a), bins, labels=labels).values.codes
        return ind, uniq

    sizes = np.unique([len(v) for v in coordinate_columns.values()])
    assert len(sizes) == 1, "All coordinates need to be the same length"

    indexes = {k: get_index_labels(v) for k, v in coordinate_columns.items()}
    labels = {k: v[1] for k, v in indexes.items()}
    sizes = {k: len(v) for k, v in labels.items()}
    indexes = {k: v[0] for k, v in indexes.items()}

    keys = list(sizes.keys())
    sizes = [len(data_columns)] + [sizes[k] for k in keys]
    indexes = [indexes[k] for k in keys]

    gridded = np.ndarray(sizes) * np.nan
    for i in range(sizes[0]):
        colidx = tuple([i] + indexes)
        return (gridded, colidx, data_columns[i])
        gridded.__setitem__(colidx, data_columns[i])

    if not return_dataaray:
        return gridded.squeeze()

    col_names = {"columns": np.arange(sizes[0]).astype("O")}
    for i, col in enumerate(data_columns):
        if hasattr(col, "name"):
            col_names["columns"][i] = col.name

    labels.update(col_names)
    da = xr.DataArray(gridded, dims=["columns"] + keys, coords=labels)

    return da.squeeze()


def colocate_dataarray(da, verbose=True, **coords):
    """
    Colocates SOCAT data with data data in a netCDF (xarray.DataArray).

    Parameters
    ----------
    da : xr.DataArray
        a dataArray that will be matched with the values in coords
    coords : array-like (float)
        keynames need to be matched with the dimension names in da.
        values need to be matched with the dtypes in da.

    Returns
    -------
    matched_values : float
        an array that matches length of socat data, but is colocated
        nc_dataarray data
    """

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def make_bins(x):
        x = np.array(x)
        if np.issubdtype(x.dtype, np.datetime64):
            dx = np.nanmean(np.diff(x).astype(int)).astype("timedelta64[ns]")
            bins = np.arange(x[0] - dx / 2, x[-1] + dx, dx, dtype="datetime64[ns]")
        else:
            dx = np.nanmean(np.diff(x))
            bins = np.linspace(x[0] - dx / 2, x[-1] + dx / 2, x.size + 1)
        assert (len(x) + 1) == len(bins), "bins must be one longer than centers"
        return bins

    dims = da.dims
    keys = list(coords.keys())

    coords = {k: np.array(v, ndmin=1) for k, v in coords.items()}

    not_matched = [k for k in keys if k not in dims]
    coord_lengths = set([coords[k].size for k in keys])

    assert len(coord_lengths) == 1, "All given coords need to be the same length"
    assert len(keys) > 0, "You need to have at least one coordinate to match"
    assert len(not_matched) == 0, f"Coords are not in da: {str(not_matched)}"

    dtypes = {k: [da[k].dtype, coords[k].dtype] for k in keys}
    type_missmatch = [k for k in keys if coords[k].dtype.kind != da[k].dtype.kind]

    assert len(type_missmatch) == 0, (
        f"Dimensions to not have the same type: "
        f"{str({k: dtypes[k] for k in type_missmatch})}"
    )

    vprint("{}:".format(da.name), end=" ")

    ranges = {k: slice(np.nanmin(coords[k]), np.nanmax(coords[k])) for k in keys}

    xdr = da.sel(**ranges)
    if xdr.size > (720 * 1440 * 365 * 2):
        raise ValueError(
            f"the range of the input coordinates is too large to load for the "
            f"given dataarray {da.name} with a shape of {da.shape}"
        )
    elif xdr.size == 0:
        vprint("no data within coordinate ranges - returning nans")
        return np.ones(list(coord_lengths)[0]) * np.NaN
    else:
        vprint("loading", end=", ")
        xdr = xdr.load()

    vprint("making bins", end=", ")
    bins = {k: make_bins(xdr[k]) for k in dims}

    # array of indicies based on cutting data
    binned = pd.DataFrame(
        {
            k: pd.cut(
                coords[k],
                bins[k],
                labels=np.arange(bins[k].size - 1, dtype=int),
            )
            for k in keys
        }
    )

    vprint("fetch data")
    # IF there were any nans, they will be negative from the int transformation
    # the bad_index array finds these and replaces them with 0 temporarily
    # they are finally replaced with nans
    null = binned.isnull().any(axis=1)
    binned = binned.fillna(0)

    index = ()
    for key in dims:
        if key in coords:
            index += (binned[key].astype(int),)
        else:
            index += (slice(None),)

    out = xdr.values.__getitem__(index).T
    out[null] = np.NaN

    return out


def grid_dataframe_to_target(
    time,
    lat,
    lon,
    cols,
    target,
    verbosity=20,
    sparse=False,
    aggregators=["mean", "std", "count"],
):
    """
    Does gridding for ungridded data to match an xarray.DataArray

    Parameters
    ----------
    time : datetime64[ns]
        time of measurements
    lat : float
        must range from -90 : 90 (will be flipped if 90 : -90)
    lon : float
        must range from -180 : 180
    target : xr.DataArray
        a regularly gridded 3D data array with dims (time, lat, lon)
        this will only be used to define the output format.

    Returns
    -------
    matched_value : float
        an array that matches length of socat data, but is colocated
        nc_dataarray data

    """
    import numpy as np

    def log(msg):
        logger.log(verbosity, msg)

    log(f"[GRID] creating bins for {target.name}")
    time_name, lat_name, lon_name = target.dims

    flipped = True if any(target[lon_name] < 0) else False
    target = target.conform.lon_0E_360E()
    lon = lon % 360

    t = target[time_name].values
    y = target[lat_name].values
    x = target[lon_name].values

    dy = np.nanmean(np.diff(y))
    dx = np.nanmean(np.diff(x))

    xbins = np.linspace(x[0] - dx / 2, x[-1] + dx / 2, x.size + 1)
    ybins = np.linspace(y[0] - dy / 2, y[-1] + dy / 2, y.size + 1)

    dt = np.nanmean(np.diff(t).astype(int)).astype("timedelta64[ns]")
    # the 1.2 is an error factor to account for sometimes the time doesnt
    # create the right number of bins
    tbins = np.arange(t[0], t[-1] + dt * 1.2, dt, dtype="datetime64[ns]")

    # removing points that are outside the time grid
    t0, t1 = tbins[[0, -1]]
    mask = (time >= t0) & (time <= t1)
    time = time[mask]
    lat = lat[mask]
    lon = lon[mask]
    cols = cols[mask]

    # array of indicies based on cutting data
    tyx = pd.DataFrame(
        np.vstack(
            [
                pd.cut(time, tbins, labels=t).astype("O"),
                pd.cut(lat, ybins, labels=y).astype("O"),
                pd.cut(lon, xbins, labels=x).astype("O"),
            ]
        ).T,
        columns=["time", "lat", "lon"],
    )
    tyx = tyx.dropna(how="any")

    df = cols.iloc[tyx.index].drop(columns=["time", "lat", "lon"], errors="ignore")
    df["time"] = tyx["time"].values
    df["lat"] = tyx["lat"].values
    df["lon"] = tyx["lon"].values

    log("[GRID] Grouping data by (time, lat, lon)")
    grp = df.groupby(["time", "lat", "lon"])
    agg = grp.aggregate(aggregators)
    if len(aggregators) > 1:
        agg.columns = ["_".join(col) for col in agg.columns.values]
    else:
        agg.columns = [col[0] for col in agg.columns.values]

    if sparse:
        log("[GRID] Output will be sparse")
        xds = xr.Dataset.from_dataframe(agg, sparse=True)
    else:
        log("[GRID] Output will be dense")
        xds = xr.Dataset.from_dataframe(agg).reindex_like(target)

    if flipped:
        log("[GRID] Longitude was flipped for gridding, flipping back to -180 : 180")
        xds = xds.conform.lon_180W_180E()

    xds["time_bounds"] = xr.DataArray(
        data=np.c_[tbins[:-1], tbins[1:]], dims=("time", "bounds"), coords={"time": t}
    )

    return xds


def _make_bins_from_gridded_coord(x):
    from .grid import estimate_grid_spacing

    x = np.array(x)
    if np.issubdtype(x.dtype, np.datetime64):
        dx = np.nanmean(np.diff(x).astype(int)).astype("timedelta64[ns]")
        bins = np.arange(x[0] - dx / 2, x[-1] + dx, dx, dtype="datetime64[ns]")
    else:
        dx = estimate_grid_spacing(x)
        bins = np.linspace(x[0] - dx / 2, x[-1] + dx / 2, x.size + 1)
    assert (len(x) + 1) == len(bins), "bins must be one longer than centers"
    return bins


@pd.core.accessor.register_dataframe_accessor("gridding")
class PandasGridder:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def grid_to_target_array(
        self, target_array, aggregators=("mean",), sparse=True, verbosity=20
    ):
        df = self._obj

        dims = ["time", "lat", "lon"]
        for key in dims:
            assert (
                key in df.columns
            ), f"{key} not in columns - required for gridding. Try resetting index."
            assert (
                key in target_array.dims
            ), f"{key} not in target_array.dims - required for gridding."
            a = df[key].values[0]
            b = target_array[key].values[0]
            a - b  # an error will be raised if they are not the same type

        t, y, x = df[dims].values.T
        cols = df.columns.drop(dims)

        t = t.astype("datetime64[ns]")
        out = grid_dataframe_to_target(
            t,
            y,
            x,
            df[cols],
            target_array,
            sparse=sparse,
            aggregators=aggregators,
            verbosity=verbosity,
        ).assign_attrs(
            history=f"[AMC] gridded flag data to xr.DataArray shape {target_array.shape}"
        )
        return out
