from functools import wraps as _wraps
from PyCO2SYS.api import CO2SYS_wrap as co2sys 
import xarray as xr


def sensitivities(ds, n_jobs=24, pardim='time', verbose=True, **kwargs):
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
        
        out = xr.concat(results, pardim).astype('float32')
    
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
        "pHinFREE":  "phfree",
        "TEMPIN": "temperature",
        "SAL": "salinity"}
    
    sensitive_keep = {   
        "gammaTCin": "gamma_dic",
        "gammaTAin": "gamma_alk",
        "omegaTCin": "omega_dic",
        "omegaTAin": "omega_alk",
        "betaTCin":  "beta_dic",
        "betaTAin":  "beta_alk"}

    variables = out[list(variables_keep)].rename(variables_keep)
    variables['hplus'] = (10**(-variables.phfree) * 1e9).assign_attrs(units='nmol/kg', description='phfree converted to H+')
    
    sensitive = out[list(sensitive_keep)].rename(sensitive_keep)
    gamma = xr.Dataset()
    gamma['dic'] = variables.dic / sensitive.gamma_dic / 1e6
    gamma['alk'] = variables.alk / sensitive.gamma_alk / 1e6
    gamma['temp'] = variables.temperature * 0 + 0.0423
    gamma['fw'] = 1 + sensitive.gamma_dic + sensitive.gamma_alk
    # omega is Aragonite sensitivity
    omega = xr.Dataset()
    omega['dic'] = variables.dic / sensitive.omega_dic / 1e6
    omega['alk'] = variables.alk / sensitive.omega_alk / 1e6
    omega['temp'] = variables.temperature * 0 + 0.0052
    omega['fw'] = 1 + sensitive.omega_dic + sensitive.omega_alk
    # beta is [H+] sensitivity
    beta = xr.Dataset()
    beta['dic'  ] = variables.dic / sensitive.beta_dic / 1e6
    beta['alk'  ] = variables.alk / sensitive.beta_alk / 1e6
    beta['temp' ] = variables.temperature * 0 + 0.0356
    beta['fw'   ] = 1 + sensitive.beta_dic + sensitive.beta_alk
    
    sensitive = xr.merge([
        gamma.to_array(dim='driver', name='gamma'),
        omega.to_array(dim='driver', name='omega'),
        beta.to_array(dim='driver', name='beta'),
    ])

    return xr.merge([variables, sensitive])


def decompose_carbsys_param_drivers(input_a, input_b, sens_name, var_name):
    """
    This may need revision

    average approach: a = averages; b = data
    trend approach:   a = data;     b = trends
        both of these should be a xr.Dataset, where variables are the 
        driver names. There must also be a component dimension that has 
        the following order (index starting 0): 
            0 = sensitivity
            1 = carbonate system parameter
            2 = change in the driver variable
            3 = driver variable (for scaling)
    sens_name: name of the sensitivity parameter (e.g. beta/gamma)
    var_name:  name of the carbonate system parameter name (e.g. Hplus/pCO2)
    """
    
    a = input_a
    b = input_b
    
    decomp = xr.Dataset()
    for key in a:
        decomp[key] = xr.concat([
            (b[key][0] * a[key][1] * a[key][2] / a[key][3]).assign_coords(component=sens_name),
            (a[key][0] * b[key][1] * a[key][2] / a[key][3]).assign_coords(component=var_name),
            (a[key][0] * a[key][1] * b[key][2] / a[key][3]).assign_coords(component='change')],
            'component')
    decomp = decomp.where(lambda x: x !=0)
    
    return decomp.to_array(dim='driver', name=var_name)
