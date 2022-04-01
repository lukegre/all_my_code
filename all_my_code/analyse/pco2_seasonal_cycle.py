"""
Designed to be a self contained script

Main functions are decompose_spco2_with_avg and decompose_spco2_with_trend
"""
import xarray as xr
import numpy as np
import joblib
import os


# SEASONAL CYCLE FITTING # 
def graven2013_seasonal_fit(da, n_jobs=24):
    """
    Fits a seasonal cycle to data using cos and sin functions. 
    Accounts for annual and 6-monthly cycles. 
    
    Fits the Graven et al. (2013)
    assumes time, lat, lon dataarray where time is a monthly resolution
    
    """
    
    def fit_graven2013_numpy(x, y):
        """
        Fits a seasonal cycle to data using cos and sin functions. 
        Accounts for annual and 6-monthly cycles. 

        Args:
            x (array-like): the time-dimension
            y (array-like): the time-varying value to fit a seasonal cycle to

        Returns:
            array-like: a fitted seasonal cycle (y-hat)
        """
        from scipy import optimize
        from numpy import sin, cos, pi, nan

        def func(t, a1, a2, a3, a4, a5, a6, a7):
            """function to fit as defined by Peter"""
            return (
                a1 + a2*t + a3*t**2 + 
                a4 * sin(2 * pi * t) + 
                a5 * cos(2 * pi * t) + 
                a6 * sin(4 * pi * t) + 
                a7 * cos(4 * pi * t))
        try:
            params, params_covariance = optimize.curve_fit(func, x, y)
        except RuntimeError:
            params = [nan] * 7
        yhat = func(x, *params)

        return yhat
    
    da_stack = da.stack(coords=['lat', 'lon'])

    Ynan = da.stack(coords=['lat', 'lon']).values.T
    Yhat = Ynan * np.nan
    
    nan_mask = np.isnan(Ynan).any(axis=1)
    Y = Ynan[~nan_mask]
    # the x-input has to be in years for the fitting function to work
    x = da.time.values.astype('datetime64[M]').astype(float) / 12
    
    n_yrs = x.size // 12
    lat = da.lat.values
    lon = da.lon.values

    pool = joblib.Parallel(n_jobs=n_jobs)
    func = joblib.delayed(fit_graven2013_numpy)
    queue = [func(x, y) for y in Y]
    Yhat[~nan_mask] = np.array(pool(queue))
    
    Yhat_clim = (
        Yhat
        .reshape(lat.size, lon.size, n_yrs, 12)
        .mean(axis=2)
        .transpose(2, 0, 1))

    djf = Yhat_clim[[-1, 0, 1]].mean(axis=0)
    jja = Yhat_clim[[5, 6, 7]].mean(axis=0)

    rmse = (
        (np.square(Yhat - Ynan).mean(axis=1)**0.5)
        .reshape(lat.size, lon.size))

    dims = 'month', 'lat', 'lon'
    ds = xr.Dataset(coords=dict(month=range(1, 13), lat=da.lat.values, lon=da.lon.values))
    ds['seasonal_clim'] = xr.DataArray(Yhat_clim, dims=dims)
    ds['seas_jja_minus_djf'] = xr.DataArray(jja - djf, dims=dims[1:]).assign_attrs(description='JJA - DFJ')
    ds['rmse'] = xr.DataArray(rmse, dims=dims[1:])

    return ds


def climatology_based_seasonal_fit(da):
    """
    Simply calculates seasonal cycle amplitude as the clim.max - clim.mean
    
    Args:
        da (xr.DataArray): input array with a time dimension that is mothly
    
    Returns:
        xr.DataArray: A single `timestep` of data
    """
    clim = da.groupby('time.month').mean('time')
    ds = xr.Dataset()
    ds['seasonal_clim'] = clim
    ds['seas_jja_minus_djf'] = (
        clim.sel(month=[6, 7, 8]).mean('month') - 
        clim.sel(month=[12, 1, 2]).mean('month'))
    ds['rmse'] = np.square(da - clim.sel(month=da.time.dt.month)).mean('time')**0.5
    
    return ds


def fit_seasonal_cycle(da, window_size_years=3, func=graven2013_seasonal_fit):
    
    amplitudes = []
    y0, y1 = da.time.dt.year[[0, -1]].values
    
    nicename = '.'.join([func.__module__, func.__name__])
    dy = int((window_size_years - 1) / 2)
    years = list(range(y0, y1+1))
    for year in years:
        print(year, end=', ')
        t0 = year - dy
        t1 = year + dy
        s0 = str(t0)
        s1 = str(t1)
        subset = da.sel(time=slice(s0, s1)).dropna('time', how='all')
        amplitudes += func(subset),
    print('')
    
    amplitude = (
        xr.concat(amplitudes, 'time')
        .assign_coords(time=[np.datetime64(f"{y}-01-01") for y in years])
        .assign_attrs(description=f'Seasonal cycle fit with {nicename}')
    )
    
    return amplitude


def create_seasonal_cycle_fit_netcdf(ds, dest='.'):
    
    ds = ds.load()
    
    name = os.path.join(dest, 'seasonal_cycle_fit_{key}_{y0}-{y1}.nc')
    os.makedirs(dest, exist_ok=True)
    y0, y1 = ds.time.dt.year[[0, -1]].values
    
    for key in ds:
        sname = name.format(key=key, y0=y0, y1=y1)
        if os.path.exists(sname):
            continue
        print(f'Working on file: {sname}')
        da = ds[key]
        # output contains seasonal cycle fits, JJA-DJF, and RMSE
        seasonal_cycle = fit_seasonal_cycle(da, window_size_years=3)
        save_nc_zip(seasonal_cycle, sname)
    
    # now working on only the JJA-DJF differences
    sname = os.path.join(dest, f'seasonal_cycl_fit_jja-djf_{y0}-{y1}.nc')
    if os.path.exists(sname):
        return xr.open_dataset(sname)
    
    delta_seas = xr.Dataset()
    for key in ds:
        fname = name.format(key=key, y0=y0, y1=y1)
        delta_seas[key] = xr.open_dataset(fname).seas_jja_minus_djf
    delta_seas = delta_seas.assign_attrs(
        description=('JJA-DJF calculated using the Graven (2013) approach'))
    
    save_nc_zip(delta_seas, sname)
    return delta_seas


def prep_data(pco2, dic, alk, temp, salt):
    
    salt_norm = salt.mean('time')
    sdic = (dic / salt * salt_norm).clip(1500, 3000).rename('s_dissicos')
    salk = (alk / salt * salt_norm).clip(1500, 3000).rename('s_talkos')
    pco2 = pco2.rename('spco2')
    temp = temp.rename('tos')
    salt = salt.rename('sos')
    
    ds = xr.merge([pco2, sdic, salk, temp, salt]).assign_attrs(
        description=(
            's_dissicos and s_talkos were normalised with the long-term '
            'mean of sos'))
    
    ds = ds.assign_coords(lon=lambda x: x.lon % 360).sortby('lon')
    ds = ds.astype('float32')
    
    return ds
    

# SENSITIVITY GAMMA # 
def revelle_factor(alk, dic):
    gamma = (
        (3 * alk * dic - 2 * dic ** 2) / 
        ((2 * dic - alk) * (alk - dic)))
    return gamma


def alkalinity_buffer_factor(alk, dic):
    gamma = -(alk ** 2) / ((2 * dic - alk) * (alk - dic))
    return gamma


def create_gamma(alk, dic, temp, dest='.'):
    
    if dest is not None:
        y0, y1 = temp.time.dt.year[[0, -1]].values
        name = os.path.join(dest, f'spco2_sensitivity_gamma_{y0}-{y1}.nc')
        file = return_ds_else_create_dir(name)
        if file is not None:
            return file
    
    gamma = xr.Dataset()
    
    gamma['s_dissicos'] = revelle_factor(alk, dic)
    gamma['s_talkos'] = alkalinity_buffer_factor(alk, dic)
    gamma['tos'] = temp * 0 + 0.0423
    gamma['freshwater'] = 1 + gamma['s_talkos'] + gamma['s_dissicos'] 
    
    if dest is not None:
        save_nc_zip(gamma, name)
    
    return gamma


# DECOMPOSITION
def make_dataset_for_decomposition(spco2, dic, alk, temp, salt, dest='.'):
    
    sname = os.path.join(dest, 'spco2_decomp_inputs.nc')
    if os.path.exists(sname):
        return xr.open_dataset(sname)
    
    os.makedirs(dest, exist_ok=True)
    
    print('Preparing data')
    ds_mon = prep_data(spco2, dic, alk, temp, salt).load()  # used for seasonal cycle
    ds_an = ds_mon.resample(time='1AS').mean()  # annual data used most of the time
    
    print('Fitting seasonal cycle')
    delta_seas = create_seasonal_cycle_fit_netcdf(ds_mon, dest=dest).load()
    
    print('Calculating sensitivities')
    gamma = create_gamma_netcdf(ds_an.s_talkos, ds_an.s_dissicos, ds_an.tos, dest=dest)
    gamma = gamma.rename(freshwater='sos')  # for loop to create data
    gamma = gamma.load()
    
    # set temp to 1 as this is not required as gamma does not change
    ds_an['tos'] = ds_an.tos * 0 + 1
    
    print('Creating input data')
    component = ['gamma', 'pco2', 'delta_seas', 'var']
    variables = ['s_dissicos', 's_talkos', 'tos', 'sos']
    dat = xr.Dataset()
    for key in variables:
        dat[key] = xr.concat(
            [gamma[key], ds_an.spco2, delta_seas[key], ds_an[key]],
            'component').assign_coords(component=component)
    dat = dat.rename(sos='freshwater')
    save_nc_zip(dat, sname)
    
    return dat
    
    
def _decompose_spco2(input_a, input_b, spco2, dest='.'):
    """
    a is the abundant input, 
    b is the shifting case
    """
    
    a = input_a
    b = input_b
    
    assert a.component.values.tolist() == ['gamma', 'pco2', 'delta_seas', 'var']
    assert b.component.values.tolist() == ['gamma', 'pco2', 'delta_seas', 'var']
    assert list(a.keys()) == ['s_dissicos', 's_talkos', 'tos', 'freshwater']
    assert list(b.keys()) == ['s_dissicos', 's_talkos', 'tos', 'freshwater']
    
    decomp = xr.Dataset()
    for key in a:
        decomp[key] = xr.concat([
            (b[key][0] * a[key][1] * a[key][2] / a[key][3]).assign_coords(component='gamma'),
            (a[key][0] * b[key][1] * a[key][2] / a[key][3]).assign_coords(component='pco2'),
            (a[key][0] * a[key][1] * b[key][2] / a[key][3]).assign_coords(component='delta_seas')],
            'component')
    decomp = decomp.where(lambda x: x !=0)
    
    delta_seas = create_seasonal_cycle_fit_netcdf(spco2.to_dataset(name='spco2'), dest=dest).spco2
    region = regions_seasonal_signal(spco2)
    sign = ((region.where(lambda x: x>0) - 1) // 2 * 2 - 1)
    weights = seaflux_weighting(dest)
    
    decomp = decomp.to_array(dim='driver', name='spco2_decomp').to_dataset()
    decomp['spco2_seas_dif'] = delta_seas
    decomp['region_mask'] = region
    decomp['sign'] = sign.assign_attrs(description='multiply to get to winter - summer')
    decomp['weights'] = weights
    
    decomp = decomp.assign_coords(driver=['sDIC', 'sAlk', 'Thermal', 'Freshwater'])
    
    return decomp


def decompose_spco2_with_avg(spco2, dic, alk, temp, salt, dest='.', overwrite=False):
    
    sname = os.path.join(dest, 'spco2_decomposed_avg.nc')
    if not overwrite and os.path.isfile(sname):
        return xr.open_dataset(sname)
    if overwrite and os.path.isfile(sname):
        os.remove(sname)
    
    inputs = make_dataset_for_decomposition(spco2, dic, alk, temp, salt, dest)
    average = inputs.mean('time')
    
    decomp = _decompose_spco2(average, inputs, spco2, dest)
    decomp['spco2_decomp_regional'] = make_regional_decomp(decomp)
    
    save_nc_zip(decomp, sname)
    
    return decomp


def decompose_spco2_with_trend(spco2, dic, alk, temp, salt, dest='.', overwrite=False):
    
    sname = os.path.join(dest, 'spco2_decomposed_trend.nc')
    if not overwrite and os.path.isfile(sname):
        return xr.open_dataset(sname)
    if overwrite and os.path.isfile(sname):
        os.remove(sname)
    
    inputs = make_dataset_for_decomposition(spco2, dic, alk, temp, salt, dest)
    trends = make_rolling_trends_netcdf(inputs, dest)
    
    decomp = _decompose_spco2(inputs, trends, spco2, dest)
    decomp['spco2_decomp_regional'] = make_regional_decomp(decomp)
    
    save_nc_zip(decomp, sname)
    
    return decomp
    

# REGIONAL FUNCTIONS AND DATASETS # 
def make_regional_decomp(decomp_out, variable='spco2_decomp'):
    
    ds = decomp_out
    weights = seaflux_weighting()
    decomp = ds[variable] * ds.sign
    regions = ds.region_mask.where(lambda x: x > 0)

    regional = regional_aggregation(decomp, regions, weights)
    
    region_names = ['NH-HL', 'NH-LL', 'SH-LL', 'SH-HL']
    regional = xr.concat(regional, 'region')
    regional = regional.assign_coords(region=region_names)
    regional = regional.assign_attrs(description='weighted averages')
    
    return regional


def regional_aggregation(xda, region_mask, weights=None, func='mean'):
    
    regional = []
    for r, da in xda.groupby(region_mask):
        da = da.unstack()
        if weights is not None:
            da = da.weighted(weights)
        da = getattr(da, func)(['lat', 'lon'])
        da = da.assign_coords(region=r)
        regional += da,
        
    regional = xr.concat(regional, 'region')
    return regional
    

def regions_seasonal_signal(pco2):
    """
    Creates a mask for pCO2 dividing the ocean into four latitudinal bands
    Norhtern/Southern Hemisphere, Low/High latitudes. 
    
    The split of the latitudes is based on the change in seasonal cycle which
    is expected to be at roughly 40deg N/S. 
    """
    def rolling_annual_avg(da, window=3):
        return da.rolling(time=window, center=True, min_periods=1).mean()
    
    pco2 = pco2.assign_coords(lon=lambda a: a.lon % 360).sortby('lon').load()
    quarterly = pco2.resample(time='1QS-Dec').mean('time')
    
    # select summer and winter data - DJF has to be shifted so that the 
    # year matches the winter year data
    DJF = rolling_annual_avg(quarterly[0::4].assign_coords(time=lambda a:a.time.dt.year+1))
    JJA = rolling_annual_avg(quarterly[2::4].assign_coords(time=lambda a:a.time.dt.year))
    seas = (JJA - DJF).mean('time')
    
    # getting the seasonal masks with certain conditions met
    NHHL = (((seas < 0) & (seas.lat >  35)) | ((seas     >   0)  & ( seas.lat >  55))) & (seas.lat <  65)
    NHLL = (( seas > 0) & (seas.lat >  10)  & ( seas.lat <  50)) | ((seas.lat >  10)   & (seas.lat <  25))
    SHLL = (( seas < 0) & (seas.lat < -10)  & ( seas.lat > -55)) | ((seas.lat < -10)   & (seas.lat > -30))
    SHHL = (  seas > 0) & (seas.lat < -30)  & ( seas.lat > -65)

    # from north to south increasing
    mask = (1*NHHL + 2*NHLL + 3*SHLL + 4*SHHL).where(seas.notnull()).fillna(0)
    mask = mask.rename('pco2_seasonal_cycle_mask')

    return mask


def seaflux_weighting(dest='.'):
    import pyseaflux
    
    sname = os.path.join(dest, 'weights_flux_based_seaflux.nc')
    
    if os.path.isfile(sname):
        return xr.open_dataarray(sname)

    ds = pyseaflux.get_seaflux_data(verbose=False).sel(wind='ERA5').drop('wind')
    mask = ds.sol_Weiss74[0].notnull()

    ds = ds[[
        'kw_scaled', 
        'sol_Weiss74', 
        'ice']]

    ds['area'] = pyseaflux.get_area_from_dataset(ds)
    ds['ice_free'] = 1 - ds.ice.fillna(0)
    ds = ds.drop('ice')

    da = ds.to_array('variable')
    var = da['variable'].values.astype(str).tolist()

    var_str = ' * '.join(var)
    da = da.prod('variable')

    weights = (
        da.load().where(mask).mean('time')
        .fillna(0)
        .assign_coords(lon=lambda a: (a.lon % 360))
        .sortby('lon')
        .rename(var_str)
        .assign_attrs(
            description=f'weighting is {var_str}'))
    
    weights = weights - weights.min()
    weights = weights / weights.max()
    weights.to_netcdf(sname)
    
    return sname


# TREND FUNCTIONS #
def make_rolling_trends_netcdf(inputs, dest='.'):
    
    sname = os.path.join(dest, 'spco2_decomp_input_trends.nc')
    file = return_ds_else_create_dir(sname)
    if file is not None:
        return file

    print('Calculating trends')
    trends = (
        calc_rolling_trend(inputs.to_array(), n_jobs=36)
        .to_dataset(dim='variable')
        .transpose('component', 'time', 'lat', 'lon'))
    
    save_nc_zip(trends, sname)
    
    return trends


def calc_trend(xda, dim='time'):
    if xda.shape[0] <= 2:
        # dummy trend
        return xda[0] * np.nan

    if xda.name is None:
        name = 'slope'
    else:
        name = xda.name + '_slope'
    
    xda = xda.assign_coords(time=lambda x: x.time.dt.year)
    trend = (
        xda.polyfit('time', 1, skipna=False)
        .polyfit_coefficients[0]
        .drop('degree')
        .rename(name)
        .assign_attrs(units='units/year')
    )
    
    return trend

     
def calc_rolling_trend(da_in, window_size=3, n_jobs=36):
    
    roll = da_in.rolling(time=window_size, min_periods=3, center=True)
    
    time = xr.concat([t for t, _ in roll], 'time')
    
    if n_jobs > 1:
        pool = joblib.Parallel(n_jobs=n_jobs)
        func = joblib.delayed(calc_trend)
        queue = [func(xda) for _, xda in roll]
        trends = pool(queue)
    else:
        func = calc_trend
        trends = [func(xda) for _, xda in roll]
        
    trends = xr.concat(trends, 'time').assign_coords(time=time)

    return trends


# FILE UTILS #
def save_nc_zip(ds, sname):
    for key in ds.data_vars:
        ds[key] = ds[key].astype('float32')
    comp = dict(complevel=4, zlib=True)
    ds.to_netcdf(sname, encoding={k: comp for k in ds})


def return_ds_else_create_dir(sname):
    from pathlib import Path
    
    sname = Path(sname)
    if sname.is_file():
        return xr.open_dataset(sname)
    else:
        sname.parent.mkdir(exist_ok=True, parents=True)