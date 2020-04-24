"""
A few tools that help when you have sparse data.
These should actually be made into accessors, but
I haven't had the time.
"""


def assparse(xds_dense):
    import xarray as xr
    import numpy as np
    import sparse

    if isinstance(xds_dense, xr.DataArray):
        xds_sparse = xr.DataArray(
            sparse.as_coo(xds_dense.values, fill_value=np.NaN),
            coords=xds_dense.coords,
            dims=xds_dense.dims,
            attrs=xds_dense.attrs,
        )
        return xds_sparse

    elif isinstance(xds_dense, xr.Dataset):
        xds_sparse = xr.Dataset()
        for key in xds_dense:
            xds_sparse[key] = assparse(xds_dense[key])
        return xds_sparse


def asdense(xds_sparse):
    import xarray as xr

    if isinstance(xds_sparse, xr.DataArray):
        dense = xds_sparse.data.todense()
        xda_dense = xds_sparse.copy()
        xda_dense.values = dense

        return xda_dense

    elif isinstance(xds_sparse, xr.Dataset):
        xds_dense = xr.Dataset()
        for key in xds_sparse:
            xds_dense[key] = asdense(xds_sparse[key])
        return xds_dense


def sparse_to_pandas(xds_sparse):
    import xarray as xr
    import pandas as pd

    if isinstance(xds_sparse, xr.DataArray):
        coo = xds_sparse.data
        out = {}
        for i, key in enumerate(xds_sparse.dims):
            out[key] = xds_sparse[key].isel(**{key: coo.coords[i]}).values
        out[xds_sparse.name] = coo.data
        ser = pd.DataFrame(out)
        ser = ser.set_index(list(xds_sparse.dims))[xds_sparse.name]
        return ser
    elif isinstance(xds_sparse, xr.Dataset):
        df = []
        for key in xds_sparse:
            df += (sparse_to_pandas(xds_sparse[key]),)
        return pd.concat(df, axis=1)


def distance(lon, lat, ref_idx=None):
    """
    Great-circle distance in m between lon, lat points.

    Parameters
    ----------
    lon, lat : array-like, 1-D (size must match)
        Longitude, latitude, in degrees.
    ref_idx : None, int
        Defaults to None, which gives adjacent distances.
        If set to positive or negative integer, distances
        will be calculated from that point

    Returns
    -------
    distance : array-like
        distance in meters between adjacent points
        or distance from reference point

    """
    import numpy as np

    lon = np.array(lon)
    lat = np.array(lat)

    earth_radius = 6371e3

    if not lon.size == lat.size:
        raise ValueError(
            "lon, lat size must match; found %s, %s" % (lon.size, lat.size)
        )
    if not len(lon.shape) == 1:
        raise ValueError("lon, lat must be flat arrays")

    lon = np.radians(lon)
    lat = np.radians(lat)

    if ref_idx is None:
        i1 = slice(0, -1)
        i2 = slice(1, None)
        dlon = np.diff(lon)
        dlat = np.diff(lat)
    else:
        ref_idx = int(ref_idx)
        i1 = ref_idx
        i2 = slice(0, None)
        dlon = lon[ref_idx] - lon
        dlat = lat[ref_idx] - lat

    a = np.sin(dlat / 2) ** 2 + np.sin(dlon / 2) ** 2 * np.cos(
        lat[i1]
    ) * np.cos(lat[i2])

    angles = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = earth_radius * angles
    d = np.r_[0, distance]

    return d
