def grid_flat_data(
    *data_columns, return_dataaray=True, **coordinate_columns
):
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
    import numpy as np
    import xarray as xr
    import pandas as pd

    def make_bins(a):
        if isinstance(a[0], np.datetime64):
            istime = True
            inf = (
                np.timedelta64(9, "Y").astype("timedelta64[ns]").astype(float)
            )
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
