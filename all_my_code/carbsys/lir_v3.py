from numpy import ndarray
import pandas as pd
import xarray as xr
from functools import wraps as _wraps


def _esper_lir(
    output_var="TA",
    depth=None,
    lat=None,
    lon=None,
    sal=None,
    temp=None,
    nitrate=None,
    oxygen=None,
    silicate=None,
):
    """
    Calculates total alkalinity using locally interpolated linear regressions
    from Carter et al. (2021).

    Note that coordinate parameters [depth, lat, lon] are not required if
    input arrays are xarray.DataArrays that contain lat and lon. If depth is
    not provided, it will be assumed to be 0 (i.e., surface).

    Parameters
    ----------
    depth : array_like
        Depth of data (m). If not provided, surface is assumed.
    lat : array_like
        Latitude of data (degrees N)
    lon : array_like
        Longitude of data (degrees E)
    sal : array-like
        salinity (always required)
    temp : array-like
        temperature in degC
    nitrate : array-like
        nitrate concentration in umol/kg
    oxygen : array-like
        oxygen concentration in umol/kg
    silicate : array-like
        silicate concentration in umol/kg

    Returns
    -------
    LIR_output : array-like
        The LIR output calculated from the input variables.

    Note
    ----
    The table below shows the technical details of which variables will
    be used in the LIR calculation. These names are consistent with the
    data that is loaded from the ESPER coefficients.

    DesiredVar   | A             B             C
    _____________|_____________________________________
    TA           | Nitrate       Oxygen        Silicate
    DIC          | Nitrate       Oxygen        Silicate
    pH           | Nitrate       Oxygen        Silicate
    phosphate    | Nitrate       Oxygen        Silicate
    nitrate      | Phosphate     Oxygen        Silicate
    silicate     | Phosphate     Oxygen        Nitrate
    O2           | Phosphate     Nitrate       Silicate

    TODO
    ----
    - return uncertainty
    - implement DIC trend
    """
    import pandas as pd
    from pandas import DataFrame
    from numpy import array, zeros, c_, inf, bool8
    from scipy.interpolate import RegularGridInterpolator as grid_interp

    if sal is None:
        raise ValueError("salinity is required")

    # find out which variables are given as input
    names = array(["sal", "temp", "nitrate", "oxygen", "silicate"])
    vars = array([sal, temp, nitrate, oxygen, silicate], dtype="O")
    avail = array([v is not None for v in vars])

    # convert all data to series if not already
    depth, lat, lon = pd.Series(depth), pd.Series(lat), pd.Series(lon)

    # make sure all variables have the same shape as salinity
    # which is the only variable that is required for all cases
    assert [
        sal.shape == v.shape for v in vars[avail]
    ], "All given inputs must be the same size"

    # find out which equation case to use
    # this is copied directly from the ESPER script
    cases = pd.DataFrame(
        {
            1: [1, 1, 1, 1, 1],
            2: [1, 1, 1, 0, 1],
            3: [1, 1, 0, 1, 1],
            4: [1, 1, 0, 0, 1],
            5: [1, 1, 1, 1, 0],
            6: [1, 1, 1, 0, 0],
            7: [1, 1, 0, 1, 0],
            8: [1, 1, 0, 0, 0],
            9: [1, 0, 1, 1, 1],
            10: [1, 0, 1, 0, 1],
            11: [1, 0, 0, 1, 1],
            12: [1, 0, 0, 0, 1],
            13: [1, 0, 1, 1, 0],
            14: [1, 0, 1, 0, 0],
            15: [1, 0, 0, 1, 0],
            16: [1, 0, 0, 0, 0],
        }
    ).T
    # matching the equation case to the input variables
    eq = (cases == avail).all(axis=1).where(lambda x: x).dropna().index.values

    # some housekeeping - make sure that the equation case is valid
    msg = "there must only be one unique equation - something wrong with cases"
    assert len(eq) == 1, msg
    eq = eq[0]  # take the first entry

    x = vars[avail].tolist() + [sal * 0 + 1]
    x = c_[[array(v, dtype=float) for v in x]].T
    x_names = names[avail].tolist() + ["0"]
    preds = DataFrame(x, columns=x_names)

    # longitude has to be 0-360
    lon360 = lon % 360
    # create coordinate df for interpolation
    coords = DataFrame(array([depth, lat, lon360]).T, columns=["depth", "lat", "lon"])

    # downloads the coefficients from the remote database
    coefs = fetch_esper_data_as_xarray(output_var).coefficients.sel(equation=eq)

    print("applying equation {}".format(eq))
    # create an empty output array
    yhat = pd.Series(
        zeros(preds.shape[0]),  # same shape as the input
        index=[depth, lat, lon],  # note we use the original lon not lon360
        name=f"{output_var}_esper",
        dtype="float32",
    )

    # create a mask for the parameter names - note that the first
    # column is the constant so will always be True, so we add that
    mask = [True] + bool8(cases.loc[eq]).tolist()
    # loop through each coeffcient and multiply it by the predictor
    # adding it to yhat. In the end we take the sum of all products
    for key in coefs.parameter.values[mask]:
        # creating a grid interpolator - very fast
        points = coefs.depth, coefs.lat, coefs.lon
        values = coefs.sel(parameter=key).values
        interpolator = grid_interp(points, values, bounds_error=False)
        # interpolate the coefficients
        coefs_i = interpolator(coords)
        # multiply the coefficients with the predictor variables
        # note we use column names so that there can't be any mixups
        yhat += coefs_i * preds.loc[:, key].values

    # interpolation sometimes results in -ve values - set those to 0
    yhat = yhat.clip(0, inf)
    # adding a description so that xarray has a bit more info
    pred_cols = str(x_names[:-1]).replace("'", "")
    yhat.description = (
        f"{output_var} estimated from the esper_lir eq #{eq} using {pred_cols}"
    )

    return yhat


def fetch_esper_data_as_xarray(lir_var):
    """
    Downloads ESPER data - all equations in one file rather than splitting
    up as in first version.
    """
    from pooch import retrieve
    from scipy.io import loadmat
    from pathlib import Path as posixpath
    from xarray import open_dataset

    path = posixpath("~/Data/cached/")
    url = f"https://github.com/BRCScienceProducts/ESPER/raw/main/ESPER_LIR_Files/LIR_files_{lir_var}_v3.mat"  # noqa
    name = posixpath(url).name

    # first we check if the processed coefficients are on disk
    sname = path.expanduser() / name.replace(".mat", ".nc")
    if sname.is_file():
        print(f"loading exising coefs for {lir_var}: {sname}")
        return open_dataset(sname)
    # if not, then we download the file and process
    else:
        fname = retrieve(str(url), None, fname=name, path=path, progressbar=True)

    coefs_columns = {
        "TA": ["0", "sal", "temp", "nitrate", "oxygen", "silicate"],
        "DIC": ["0", "sal", "temp", "nitrate", "oxygen", "silicate"],
        "pH": ["0", "sal", "temp", "nitrate", "oxygen", "silicate"],
        "phosphate": ["0", "sal", "temp", "nitrate", "oxygen", "silicate"],
        "nitrate": ["0", "sal", "temp", "phosphate", "oxygen", "silicate"],
        "silicate": ["0", "sal", "temp", "phosphate", "oxygen", "nitrate"],
        "oxygen": ["0", "sal", "temp", "phosphate", "nitrate", "silicate"],
    }

    # open the mat file
    lirs = loadmat(fname)
    grid = lirs["GridCoords"][:, :3]
    coefs = lirs["Cs"]

    index = pd.MultiIndex.from_arrays(grid.T, names=["lon", "lat", "depth"])
    rename = {i: coefs_columns[lir_var][i] for i in range(6)}

    ds = []
    for i in range(coefs.shape[-1]):
        ds += (
            (
                pd.DataFrame(coefs[:, :, i], index=index)
                .rename(columns=rename)
                .to_xarray()
                .to_array(dim="parameter", name=i + 1)
            ),
        )

    ds = (
        xr.merge(ds)
        # .chunk(dict(parameter=6))
        # filling gaps of the coarse data with nearest neighbour
        .interpolate_na(dim="lon", method="nearest")
        # forward-filling for edge case scenarios
        .ffill("lon")
        # back-filling for edge case scenarios
        .bfill("lon")
        .to_array(dim="equation", name="coefficients")
        .transpose("equation", "parameter", "depth", "lat", "lon")
        .to_dataset()
    )

    enc = {k: {"dtype": "float32", "zlib": True, "complevel": 4} for k in ds}
    ds.to_netcdf(sname, encoding=enc)

    return ds


@_wraps(_esper_lir)
def esper_lir(output_var="TA", **kwargs):
    """
    Wrapper for applying LIR functions. Checks if lat, lon, depth
    are in the dataset as coordinates. If not, then then they must
    be explicity passed as arguments.
    """
    from xarray import DataArray
    from pandas import Series

    keys = ["sal", "temp", "nitrate", "oxygen", "silicate"]

    dataarrays = dict()
    arraylike = dict()
    for key in keys:
        if key in kwargs:
            if isinstance(kwargs[key], (DataArray)):
                dataarrays[key] = kwargs[key]
            if isinstance(kwargs[key], (list, tuple, ndarray, Series)):
                arraylike[key] = kwargs[key]
    if (len(arraylike) > 0) and (len(dataarrays) > 0):
        raise ValueError(
            "data variables have to be either DataArray or array-like, not both"
        )

    # if all data kwargs are arraylike, then pass directly to _esper_lir
    if len(arraylike) > 0:
        return _esper_lir(output_var=output_var, **kwargs)
    else:
        assert all([k in kwargs["sal"].coords for k in ["lat", "lon"]])
        dataarrays = {k: dataarrays[k].conform() for k in dataarrays}
        ds = xr.merge([dataarrays[k].rename(k) for k in dataarrays])
        df_orig = ds.to_dataframe()
        df = df_orig.reset_index()
        if "depth" not in df:
            df["depth"] = 0
        required = ["lat", "lon", "depth"]
        to_drop = list(set(df.columns.tolist()) - set(required) - set(keys))
        df = df.drop(to_drop, axis=1)

        lir_ready = {k: df[k] for k in df}

        lir_output = _esper_lir(output_var=output_var, **lir_ready)
        out = lir_output.set_axis(df_orig.index).to_xarray()
        out = out.conform().assign_attrs(
            source="https://github.com/BRCScienceProducts/ESPER",
            reference="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020GB006623",
            description=lir_output.description,
        )

        return out
