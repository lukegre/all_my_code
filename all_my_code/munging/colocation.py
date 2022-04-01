import numpy as np
import pandas as pd
import xarray as xr


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
    xda = xr.DataArray(gridded, dims=["columns"] + keys, coords=labels)

    return xda.squeeze()


def colocate_xdarray(xda, verbose=True, **coords):
    """
    Colocates SOCAT data with data data in a netCDF (xarray.DataArray).

    Parameters
    ----------
    xda : xr.DataArray
        a dataArray that will be matched with the values in coords
    coords : array-like (float)
        keynames need to be matched with the dimension names in xda.
        values need to be matched with the dtypes in xda.

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

    dims = xda.dims
    keys = list(coords.keys())

    coords = {k: np.array(v, ndmin=1) for k, v in coords.items()}

    not_matched = [k for k in keys if k not in dims]
    coord_lengths = set([coords[k].size for k in keys])

    assert len(coord_lengths) == 1, "All given coords need to be the same length"
    assert len(keys) > 0, "You need to have at least one coordinate to match"
    assert len(not_matched) == 0, f"Coords are not in xda: {str(not_matched)}"

    dtypes = {k: [xda[k].dtype, coords[k].dtype] for k in keys}
    type_missmatch = [k for k in keys if coords[k].dtype.kind != xda[k].dtype.kind]

    assert len(type_missmatch) == 0, (
        f"Dimensions to not have the same type: "
        f"{str({k: dtypes[k] for k in type_missmatch})}"
    )

    vprint("{}:".format(xda.name), end=" ")

    ranges = {k: slice(np.nanmin(coords[k]), np.nanmax(coords[k])) for k in keys}

    xdr = xda.sel(**ranges)
    if xdr.size > (720 * 1440 * 365 * 2):
        raise ValueError(
            f"the range of the input coordinates is too large to load for the "
            f"given dataarray {xda.name} with a shape of {xda.shape}"
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
                coords[k], bins[k], labels=np.arange(bins[k].size - 1, dtype=int),
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
