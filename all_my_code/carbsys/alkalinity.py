from ..utils import xarray_dataset_to_column_input as _xarray_dataset_to_column_input


@_xarray_dataset_to_column_input
def calc_lee2006(lat, lon, temp, salt, return_regions=False):
    """
    Calculates alkalinity from surface temperature and salinity given
    latitude and longitude.

    For more details see Lee et al. (2006) [1]_. Coefficients can be found
    in their publication.

    Parameters
    ----------
    lat : np.array
        latitude ranging from -90 to 90
    lon : np.array
        longitude ranging from -180 to 180
    temp : np.array
        temperature in degrees C
    salt : np.array
        salinity in PSU
    return_regions : bool=False
        if set True returns the regions for the predictions and
        results are not returned

    Returns
    -------
    np.array
        alkalinity as estiamted from inputs
    np.array
        regions (as float) as defined by the input map if return_regions=True

    Note
    ----
    The ranges of the original Lee et al. (2006) paper have been extended so that
    regions that do not match their criteria are not left blank. These include
    the:
        - max salinity of the Northern Pacific: 35 --> 38
        - max salinity of the Southern Ocean: 36 --> 38
        - min salinity of the Sub Tropics: 31 --> 25 (includes Amazon and Bay of Bengal)

    References
    ----------
    .. [1] Lee, K., Tong, L. T., Millero, F. J., Sabine, C. L., Dickson, A. G.,
        Goyet, C., … Key, R. M. (2006). Global relationships of total alkalinity
        with salinity and temperature in surface waters of the world’s oceans.
        Geophysical Research Letters, 33(19), L19605.
        https://doi.org/10.1029/2006GL027207
    """

    def _get_lee_region(lat, lon, salt, temp):
        import numpy as np

        def is_eqPac(yA, xA):  # finds the equatorial pacific region
            from warnings import filterwarnings

            filterwarnings("ignore", category=UserWarning)
            # north south / east west boundaries
            yN = 20
            yS = -yN
            p0 = (yA < yN) & (yA > yS) & (xA > -140) & (xA < -75)

            # southern diagonal boundary
            x1, y1 = -140, -10
            x2, y2 = -110, yS
            # create two vectors and then calculate the cross product
            v1 = np.array([x2 - x1, y2 - y1])
            v2 = np.array([x2 - xA, y2 - yA])
            # if p1 is negative it is in EQ Pac
            p1 = (v1[0] * v2[1] - v1[1] * v2[0]) < 0

            # northern diagonal boundary
            x1, y1 = -140, 10
            x2, y2 = -110, yN
            # create two vectors and then calculate the cross product
            v1 = np.array([x2 - x1, y2 - y1])
            v2 = np.array([x2 - xA, y2 - yA])
            # if p2 is positive it is in EQ Pac
            p2 = (v1[0] * v2[1] - v1[1] * v2[0]) > 0

            return p0 * p1 * p2

        eq_pacifc = is_eqPac(lat, lon)

        region_conditions = {
            # standard definitions and then extra-* for waters that extend
            # beyond the geographical boundaries
            "EQ": lambda y, x, s, t: eq_pacifc & (t > 18) & (s >= 29) & (s <= 36.5),
            # extra-subtropical lies in the high lat extremes, meets
            # the conditions that don't match the SO (t > 20)
            "ST": lambda y, x, s, t: (y >= -30)
            & (y <= 30)
            & (t > 18)
            & (s > 25)
            & (s <= 38),
            "EST": lambda y, x, s, t: ((y >= 30) | (y <= -30))
            & (t > 18)
            & (s > 25)
            & (s <= 40),
            # extra-southernocean lies in the low lats, but meets the conditions
            # that don't match the subtropics (t < 20)
            "SO": lambda y, x, s, t: (y > -80)
            & (y < -30)
            & (t <= 20)
            & (s > 31)
            & (s <= 38),
            "ESO": lambda y, x, s, t: ((y > -30) & (y < 0))
            & (t <= 20)
            & (s > 31)
            & (s <= 38),
            # extra-northatlantic lies in the high lats, but meets the conditions
            # that don't match the subtropics (t < 20)
            "NA": lambda y, x, s, t: (y > 30)
            & (y < 80)
            & (x > -90)
            & (x < 75)
            & (t > -0.5)
            & (t <= 20)
            & (s > 31)
            & (s <= 37),
            "ENA": lambda y, x, s, t: (y > 20)
            & (y < 30)
            & (x > -90)
            & (x < 75)
            & (t > -0.5)
            & (t <= 20)
            & (s > 31)
            & (s <= 37),
            # extra-northpacific lies in the lower lats, but meets the conditions
            # that don't match the subtropics (t < 20)
            "NP": lambda y, x, s, t: (y > 30)
            & (y < 80)
            & ((x > 120) | (x < -105))
            & (t > -0.5)
            & (t <= 20)
            & (s > 31)
            & (s <= 38),
            "ENP": lambda y, x, s, t: (y > 0)
            & (y <= 30)
            & ((x > 120) | (x < -105))
            & (t > -0.5)
            & (t <= 20)
            & (s > 31)
            & (s <= 38),
        }

        regions = np.zeros([lon.size]).astype(int)
        regions[region_conditions["SO"](lat, lon, salt, temp)] = 3
        regions[region_conditions["ESO"](lat, lon, salt, temp)] = 3
        regions[region_conditions["NA"](lat, lon, salt, temp)] = 4
        regions[region_conditions["ENA"](lat, lon, salt, temp)] = 4
        regions[region_conditions["NP"](lat, lon, salt, temp)] = 5
        regions[region_conditions["ENP"](lat, lon, salt, temp)] = 5
        regions[region_conditions["ST"](lat, lon, salt, temp)] = 2
        regions[region_conditions["EST"](lat, lon, salt, temp)] = 2
        regions[region_conditions["EQ"](lat, lon, salt, temp)] = 1

        return regions

    import numpy as np
    from warnings import filterwarnings

    filterwarnings("ignore", category=RuntimeWarning)

    assert all(
        [(a is not None) for a in [lat, lon, salt, temp]]
    ), "lat, lon, salt, temp must be provided"

    lat, lon, salt, temp = [np.array(a) for a in [lat, lon, salt, temp]]

    coeffs = np.array(
        [  # the coefficients for vectorised MLR for each region
            # ones  salt35  salt35^2  temp20  temp20^2  temp29  temp29^2
            [0, 0, 0, 0, 0, 0, np.nan],  # undefined becomes nan
            [2294, 64.88, 0.39, 0, 0, -4.52, 0.232],  # pacific equatorial
            [2305, 58.66, 2.32, -1.41, 0.040, 0, 0],  # subtropics
            [2305, 53.97, 2.74, -1.16, 0.040, 0, 0],  # north atlantic
            [2305, 53.97, 2.74, -1.16, 0.040, 0, 0],  # north pacific
            [2305, 52.48, 2.85, -0.49, 0.086, 0, 0],
        ]
    )  # southern ocean

    # define the required columns for the regression
    ones = np.ones_like(lon)
    s35 = salt - 35
    t20 = temp - 20
    t29 = temp - 29  # used for the equatorial region
    X = np.array([ones, s35, s35**2, t20, t20**2, t29, t29**2]).T

    regions = _get_lee_region(lat, lon, salt, temp)
    if return_regions:
        return regions

    yhat = np.ndarray(lon.size)
    for r in range(6):
        i = regions == r
        yhat[i] = (X[i] * coeffs[r]).sum(1)

    return yhat


def esper_liar(
    depth, lat, lon, sal=None, temp=None, nitrate=None, oxygen=None, silicate=None
):
    """
    Calculates total alkalinity using locally interpolated linear regressions

    Parameters
    ----------
    depth : array_like
        Depth of data (m)
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
    alkalinity : array-like
        total alkalinity
    """
    import pandas as pd
    from pandas import concat
    from numpy import array, zeros
    from scipy.interpolate import RegularGridInterpolator as grid_interp

    def fetch_esper_data_as_xarray(eq_case):
        """
        Get the data from the remote LIAR3 database

        Parameters
        ----------
        eq_case : str
            Loads the equation case from the LIAR3 database

        Returns
        -------
        xr.DataArray
            Coefficients for the equation case
        """
        from numpy import c_ as concat
        from pooch import retrieve
        from scipy.io import loadmat
        from pathlib import Path as posixpath
        from xarray import open_dataarray

        path = posixpath("~/Data/cached/")
        url = "https://github.com/BRCScienceProducts/ESPER/raw/main/ESPER_LIR_Files/LIR_files_TA_v3.mat"  # noqa
        name = posixpath(url).name

        # first we check if the processed coefficients are on disk
        sname = path.expanduser() / name.replace(".mat", f"_eq{eq_case:02d}.nc")
        if sname.is_file():
            print(f"loading exising coefs for eq. {eq_case}: {sname}")
            return open_dataarray(sname)
        # if not, then we download the file and process
        else:
            fname = retrieve(str(url), None, fname=name, path=path, progressbar=True)

        # open the mat file
        lirs = loadmat(fname)
        grid = lirs["GridCoords"][:, :3]
        coefs = lirs["Cs"][:, :, eq_case - 1]

        data = concat[grid, coefs]
        # these names are consistent with the input names
        # and allows us to not mix up columns since we
        # use names instead :) the order is important
        names = [
            "lon",
            "lat",
            "depth",
            "0",
            "sal",
            "temp",
            "nitrate",
            "oxygen",
            "silicate",
        ]

        # coords in ESPER are in longitude, latitude, depth order
        df = pd.DataFrame(data, columns=names).set_index(["lon", "lat", "depth"])

        # now we create an xarray array
        coefs = (
            df.to_xarray()
            .transpose("depth", "lat", "lon")
            .to_array(name="esper_liar3", dim="coef")
            # filling gaps of the coarse data with nearest neighbour
            .interpolate_na(dim="lon", method="nearest")
            # backfilling for edge case scenarios
            .ffill("lon")
        )

        # drop coefs that are all nans
        keep = coefs.notnull().any(["lat", "lon", "depth"])
        coefs = coefs.where(keep).dropna(dim="coef", how="all")

        # save the coefs to speed up processing
        coefs_ds = coefs.to_dataset()
        encoding = {k: {"zlib": True} for k in coefs_ds.data_vars}
        coefs_ds.to_netcdf(sname, encoding=encoding)

        return coefs

    if sal is None:
        raise ValueError("salinity is required")

    # find out which variables are given as input
    names = array(["sal", "temp", "nitrate", "oxygen", "silicate"])
    vars = array([sal, temp, nitrate, oxygen, silicate], dtype="O")
    avail = array([v is not None for v in vars])

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
    x_names = names[avail].tolist() + ["0"]
    preds = concat(x, axis=1).set_axis(x_names, axis=1, inplace=False)

    lon360 = lon % 360
    coords = concat([depth, lat, lon360], axis=1)

    coefs = fetch_esper_data_as_xarray(eq)

    # create an empty output array
    yhat = pd.Series(
        zeros(preds.shape[0]),
        index=[depth, lat, lon],
        name="alkalinity_esper",
        dtype="float64",
    )

    for key in coefs.coef.values:
        # creating a grid interpolator - very fast
        points = coefs.depth, coefs.lat, coefs.lon
        values = coefs.sel(coef=key).values
        interpolator = grid_interp(points, values, bounds_error=False)
        # interpolate the coefficients
        coefs_i = interpolator(coords)
        # multiply the coefficients with the predictor variables
        # note we use column names so that there can't be any mixups
        yhat += coefs_i * preds.loc[:, key].values
    # now make remaining 0's NaN
    yhat = yhat.where(yhat > 0)

    return yhat


def esper_liar_xarray(
    sal=None, temp=None, nitrate=None, oxygen=None, silicate=None, depth=0
):
    # TODO: integrate the xarray and pandas versions
    """
    Calculates total alkalinity using locally interpolated linear regressions

    Parameters
    ----------
    sal : xr.DataArray
        salinity
    temp : xr.DataArray
        temperature
    nitrate : xr.DataArray
        nitrate concentration
    oxygen : xr.DataArray
        oxygen concentration
    silicate : xr.DataArray
        silicate concentration
    depth : float
        depth at which the alkalinity is calculated - only single
        depth is supported at the moment

    Returns
    -------
    alkalinity : xr.DataArray
        total alkalinity
    """
    import pandas as pd
    from ..files.download import download_file
    from ..datasets.masks import reccap2_regions as get_reccap2_regions
    from xarray import Dataset, concat
    from numpy import array

    def get_liar3_data_as_xarray(eq_case):
        """
        Get the data from the remote LIAR3 database

        Parameters
        ----------
        eq_case : str
            Loads the equation case from the LIAR3 database

        Returns
        -------
        xr.DataArray
            Coefficients for the equation case
        """
        from numpy import c_ as concat
        from scipy.io import loadmat

        url = "https://github.com/BRCScienceProducts/ESPER/raw/main/ESPER_LIR_Files/LIR_files_TA_v3.mat"  # noqa
        fname = download_file(url, path="~/Data/cached/")

        lirs = loadmat(fname)

        grid = lirs["GridCoords"][:, :3]
        coefs = lirs["Cs"][:, :, eq_case - 1]

        data = concat[grid, coefs]
        names = [
            "lon",
            "lat",
            "depth",
            "0",
            "sal",
            "temp",
            "nitrate",
            "oxygen",
            "silicate",
        ]
        df = pd.DataFrame(data, columns=names).set_index(["lon", "lat", "depth"])
        coefs = df.to_xarray().conform.transpose_dims().to_array(name="liar3_coefs")

        return coefs

    if sal is None:
        raise ValueError("salinity is required")

    names = array(["sal", "temp", "nitrate", "oxygen", "silicate"])
    vars = array([sal, temp, nitrate, oxygen, silicate], dtype="O")
    avail = array([v is not None for v in vars])

    assert [
        sal.shape == v.shape for v in vars[avail]
    ], "All given inputs must be the same size"

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

    eq = (cases == avail).all(axis=1).where(lambda x: x).dropna().index.values
    assert (
        len(eq) == 1
    ), "there must only be one unique equation - something wrong with cases"
    eq = eq[0]

    x = vars[avail].tolist() + [sal * 0 + 1]
    x_names = names[avail].tolist() + ["0"]
    preds = concat(x, dim="variable").assign_coords(variable=x_names)

    coarse = Dataset()
    coarse["coefs"] = get_liar3_data_as_xarray(eq).sel(depth=depth, method="nearest")
    coarse["mask"] = coarse.coefs.notnull().any("variable")

    reccap2 = get_reccap2_regions(resolution=1)
    fine_mask = reccap2.open_ocean > 0

    coefs = coarse.grid.regrid(mask=fine_mask).coefs

    est = (
        (coefs * preds)
        .sum("variable")
        .where(lambda x: x > 10)  # set a basic constraint
        .conform.transpose_dims()  # making sure dataset is in the right shape
        .assign_attrs(
            source="https://github.com/BRCScienceProducts/ESPER",
            reference="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020GB006623",
            description=(
                "Total alkalinity estimated from the " "esper_lir equation #{} using {}"
            ).format(eq, names[avail]),
        )
    )

    return est
