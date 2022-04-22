import xarray as xr


def sensitivities(ds, n_jobs=24, pardim="time", verbose=True, **kwargs):
    """
    Calculate the sensitivities of the CO2 system.

    Runs in parallel if n_jobs > 1 along the given dimension [time].

    Parameters
    ----------
    ds: xr.Dataset
        A dataset containing the following variables:
        - carbsys x 2 (eg. alk, dic, pco2),
        - temp_in, sal, po4, si
    n_jobs: int
        Number of cores to use.
    pardim: str
        Dimension to use for parallelization.
    verbose: bool
        Print progress of the parallel progress or CO2SYS_wrap
    **kwargs:
        Keyword arguments are passed to PyCO2SYS.api.CO2SYS_wrap.

    Returns
    -------
    ds: xr.Dataset
        A dataset containing the following variables: alk, dic,
        aragonite, calcite, pco2, phfree, temperature, salinity,
        gamma, beta, omega. The sensitivities have a driver dimension
        with coordinates: dic, alk, temp, fw (freshwater).
    """
    from joblib import delayed, Parallel

    if n_jobs == 1:
        kwargs.update(verbose=verbose)
        return _sensitivities(ds, **kwargs)
    else:
        kwargs.update(verbose=False)
        func = delayed(_sensitivities)
        pool = Parallel(n_jobs=n_jobs, verbose=verbose)

        size = ds[pardim].size
        queue = [func(ds.isel(**{pardim: [i]}), **kwargs) for i in range(size)]

        results = pool(queue)

        out = xr.concat(results, pardim).astype("float32")

    return out


def _sensitivities(ds_co2sys_inputs, **kwargs):
    """
    A wrapper around PyCO2SYS.api.CO2SYS_wrap to calculate the
    marine carbonate system parameter senstivities:
        beta = H+
        gamma = pCO2
        omega = OmegaAR/CA

    Parameters
    ----------
    ds_co2sys_inputs: xr.Dataset
        Names of the variables must match the inputs
        to PyCO2SYS.api.CO2SYS_wrap (see for help).
    **kwargs:
        Keyword arguments are passed to PyCO2SYS.api.CO2SYS_wrap.

    Returns
    -------
    ds: xr.Dataset
        A dataset containing the following variables: alk, dic,
        aragonite, calcite, pco2, phfree, temperature, salinity,
        gamma, beta, omega. The sensitivities have a driver dimension
        with coordinates: dic, alk, temp, fw (freshwater).
    """
    from PyCO2SYS.api import CO2SYS_wrap as co2sys

    ds = ds_co2sys_inputs
    co2sys_inputs = {k: ds[k] for k in ds}

    out = co2sys(**co2sys_inputs, **kwargs)

    variables_keep = {
        "TAlk": "alk",
        "TCO2": "dic",
        "OmegaARin": "aragonite",
        "OmegaCAin": "calcite",
        "pCO2in": "pco2",
        "pHinFREE": "phfree",
        "TEMPIN": "temperature",
        "SAL": "salinity",
    }

    sensitive_keep = {
        "gammaTCin": "gamma_dic",
        "gammaTAin": "gamma_alk",
        "omegaTCin": "omega_dic",
        "omegaTAin": "omega_alk",
        "betaTCin": "beta_dic",
        "betaTAin": "beta_alk",
    }

    variables = out[list(variables_keep)].rename(variables_keep)
    variables["hplus"] = (10 ** (-variables.phfree) * 1e9).assign_attrs(
        units="nmol/kg", description="phfree converted to H+"
    )

    sensitive = out[list(sensitive_keep)].rename(sensitive_keep)
    gamma = xr.Dataset()
    gamma["dic"] = variables.dic / sensitive.gamma_dic / 1e6
    gamma["alk"] = variables.alk / sensitive.gamma_alk / 1e6
    gamma["temp"] = variables.temperature * 0 + 0.0423
    gamma["fw"] = 1 + sensitive.gamma_dic + sensitive.gamma_alk
    # omega is Aragonite sensitivity
    omega = xr.Dataset()
    omega["dic"] = variables.dic / sensitive.omega_dic / 1e6
    omega["alk"] = variables.alk / sensitive.omega_alk / 1e6
    omega["temp"] = variables.temperature * 0 + 0.0052
    omega["fw"] = 1 + sensitive.omega_dic + sensitive.omega_alk
    # beta is [H+] sensitivity
    beta = xr.Dataset()
    beta["dic"] = variables.dic / sensitive.beta_dic / 1e6
    beta["alk"] = variables.alk / sensitive.beta_alk / 1e6
    beta["temp"] = variables.temperature * 0 + 0.0356
    beta["fw"] = 1 + sensitive.beta_dic + sensitive.beta_alk

    sensitive = xr.merge(
        [
            gamma.to_array(dim="driver", name="gamma"),
            omega.to_array(dim="driver", name="omega"),
            beta.to_array(dim="driver", name="beta"),
        ]
    )

    return xr.merge([variables, sensitive])


def decompose_carbsys(
    variable, sensitivity, scaling, driver_change, time_dim="time", with_slope=False
):
    """
    Decompose a carbonate system variable into driver and mechanism components.

    A taylor decomposition

    Note
    ----
    All inputs except 'variable' should contain the dimension 'driver'
    which must contain: dic, alk, temp, and sal

    Parameters
    ----------
    variable : xarray.DataArray
        the marine carbonate system variable to decompose. Could be
        pCO2, [H+], or Omega Ar/Ca
    sensitivity : xarray.DataArray
        the sensitivity of the variable to drivers. The sensitivity
        should be gamma (pCO2), beta (H+), or omega (ar/ca). Should
        contain a sensitivity for each driver (see note above)
    scaling : xarray.DataArray
        the driver variables that will be used for scaling
    driver_change : xarray.DataArray
        the change in the drivers. Can be the derivative of the
        change or the change itself if you'd like to see the temporal
        component - note that you will have to calculate the slope
        of the changes to get the true contribution of the driver
        change / time step.

    Returns
    -------
    xarray.DataArray
        the decomposed variable with the driver and mechanism.
        drivers: are dic, alk, temp, and sal
        mechanisms are: sensitivity, driver_change, variable_change
    """
    from ..stats.time_series import slope
    from numpy import isclose

    drivers = ["dic", "alk", "temp", "sal"]
    check_dims_in_da = lambda x: all([d in x.driver for d in drivers])

    msg = f"{{}} must contain the dimension 'driver' which must contain: {drivers}"
    assert check_dims_in_da(sensitivity), msg.format("sensitivity")
    assert check_dims_in_da(scaling), msg.format("scaling")
    assert check_dims_in_da(driver_change), msg.format("driver_change")

    t = variable.time
    compare_times = lambda x: t.size == x.time.size
    assert compare_times(sensitivity), "time must be same for variable and sensitivity"
    assert compare_times(scaling), "time must be the same for variable and scaling"
    assert isclose(
        t.size, driver_change.time.size, rtol=0.2
    ), "time must be similar for variable and driver_change"

    assert all(
        t == sensitivity.time
    ), "time must be the same for variable and sensitivity"
    assert all(t == scaling.time), "time must be the same for variable and scaling"
    assert all(
        t == driver_change.time
    ), "time must be the same for variable and driver_change"

    variable = variable.broadcast_like(sensitivity)
    scaling = scaling.where(lambda x: x.driver != "temp").fillna(1)

    # building the dataset that will be used for the taylor decomposition
    mechanisms = ["sensitivity", "variable", "driver_change", "scaling"]
    objs = [sensitivity, variable, driver_change, scaling]
    mech = xr.IndexVariable("mechanism", mechanisms)
    dat = xr.concat(objs, dim=mech).to_dataset(dim="driver")

    # this removes the nans that result from any shorter time series
    dat = dat.dropna(dim="time", how="any")

    a = dat.mean(time_dim, keepdims=0)
    b = dat.map(slope, dim=time_dim) if with_slope else dat

    decomp = _taylor_decomposition_carbsys(a, b, sensitivity.name, variable.name)

    return decomp


def _taylor_decomposition_carbsys(input_a, input_b, sens_name, var_name):
    """
    Decompose a carbonate system variable into driver and mechanism components.

    Mechanisms are a dimension of the both datasets with the following
    order: sensitivity, variable, driver_change, scaling
    Parameters
    ----------
    input_a : xarray.Dataset
        dataset variables are the drivers and the mechanisms
        are are the first dimension of the dataset
    input_b : xarray.Dataset
        the slope/ of the drivers and mechanisms. Dataset vars
        are the drivers and the metchanisms are are the a dimension
        first dimension of the dataset
    sens_name : str
        the name of the sensitivity variable (e.g. beta/gamma)
    var_name : str
        the name of the variable (e.g. Hplus/pCO2)

    Returns
    -------
    xarray.DataArray
        the decomposed variable with the driver and mechanisms as dimensions
    """

    a = input_a
    b = input_b

    decomp = xr.Dataset()
    for key in a:
        decomp[key] = xr.concat(
            [  # sensitiv.   variable     change      scaling
                (b[key][0] * a[key][1] * a[key][2] / a[key][3]).assign_coords(
                    mechanism=sens_name
                ),
                (a[key][0] * b[key][1] * a[key][2] / a[key][3]).assign_coords(
                    mechanism=var_name
                ),
                (a[key][0] * a[key][1] * b[key][2] / a[key][3]).assign_coords(
                    mechanism="driver_change"
                ),
            ],
            "mechanism",
        )
    decomp = decomp.where(lambda x: x != 0)

    return decomp.to_array(dim="driver", name=var_name)
