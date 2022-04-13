
from ..files.download import download_file
import xarray as xr
import os


def socat_gridded(version='2021', save_dir='~/Data/cached/', delete_intermediate_files=True):
    from pathlib import Path as posixpath

    url = f"https://www.socat.info/socat_files/v{version}/SOCATv{version}_tracks_gridded_monthly.nc.zip"
    fname = posixpath(url).name
    zpath = os.path.join(os.path.expanduser(save_dir), fname)
    fpath = os.path.join(os.path.expanduser(save_dir), fname.replace('.zip', ''))

    if not os.path.exists(fpath):
        dpath = download_file(url, save_dir, progress=True)
        ds = xr.open_dataset(dpath).drop('tmnth_bnds')
        ds.to_netcdf_with_compression(fpath)
    
        if delete_intermediate_files:
            os.remove(dpath)
            os.remove(zpath)
            posixpath(dpath).parent.rmdir()
    
    return xr.open_dataset(fpath, chunks={}).conform()


def _download_ethz_data(save_dir='~/Data/cached/', version='2021'):
    """
    Downloads ETHZ data from the NOAA-NODC data archives
    
    Parameters
    ----------
    save_dir: path string
        A path to where the data will be downloaded. Will be expanded so ~ 
        and relative paths can be used. Defaults to HOME/Downloads
    version: str[2021]
        The version of ETHZ-OceanSODA to download. Supports 2020, 2021
    
    Returns
    -------
    str: 
        the name of the downloaded files (not a xr.Dataset)
    """
    from pooch import retrieve, HTTPDownloader
    from os.path import expanduser, abspath
    
    save_dir = expanduser(save_dir)
    save_dir = abspath(save_dir)
    
    url = "https://www.nodc.noaa.gov/archive/arc0160/0220059/"
    if str(version) == '2020':
        fname = "OceanSODA-ETHZ_1985-2019_v2020b.nc"
        url += f"2.2/data/0-data/{fname}"
    elif str(version) == '2021':
        fname = "OceanSODA-ETHZ_GRaCER_v2021a_1982-2020.nc"
        url += f"3.3/data/0-data/{fname}"
    
    name = retrieve(
        url, None, fname, path=save_dir, 
        downloader=HTTPDownloader(progressbar=True))
    
    return name


def oceansoda_ethz(save_dir='~/Data/cached/', version='2021'):
    """
    Downloads and homogenises variable names for the different ETHZ versions 
    (2020, 2021). Names are changed to match the v2021 output
    
    Downloads are made using pooch.retrieve. Will not be downloaded if the data
    already exists at the target location. 
    
    Parameters
    ----------
    save_dir: path string
        A path to where the data will be downloaded. Will be expanded so ~ 
        and relative paths can be used. Defaults to HOME/Downloads
    version: str[2021]
        The version of ETHZ-OceanSODA to download. Supports 2020, 2021
    
    Returns
    -------
    xr.DatasetA homogenized 
    
    """
    
    fname = _download_ethz_data(save_dir, version)
    ds = xr.open_dataset(fname)
    
    # unifying 2020 to 2021 naming (2020 names are not same as 2021)
    if str(version) == '2020':
        unified_names = dict(
            DIC='dic',
            TA='alk',
            pCO2='spco2',
            pH='ph_total',
            HCO3='hco3',
            CO3='co3',
            omegaAR='omega_ar',
            omegaCA='omega_ca',
            TAstd='talk_uncert',
            pCO2std='spco2_uncert')

    elif str(version) == '2021':
        unified_names = dict(
            talk='alk',
            talk_uncert='alk_uncert'
        )
    
    ds = ds.rename(unified_names)
    
    print('[H+], sDIC and sALK calculated (normed to local long-term mean)')
    ds['h'] = (10**(-ds.ph_total) * 1e9).assign_attrs(units='nmol/kg')
    ds['sdic'] = ds.dic / ds.salinity * ds.salinity.mean('time')
    ds['salk'] = ds.alk / ds.salinity * ds.salinity.mean('time')

    return ds


def seaflux(var_name='pco2atm', save_dir='~/Data/cached/'):
    """
    Downloads the SeaFlux dataset from Zenodo

    Parameters
    ----------
    var_name: str | list
        The variable to download: fgco2, spco2_unfilled, spco2_filler,
        pco2atm, area, ice, sol, kw
    save_dir: path string
        A path to where the data will be downloaded. Will be expanded so ~
        and relative paths can be used. Defaults to HOME/Downloads
    
    Returns
    -------
    xr.Dataset: 
        The downloaded dataset
    """
    import numpy as np

    variables = dict(
        fgco2="https://zenodo.org/record/5482547/files/SeaFlux_v2021.04_fgco2_all_winds_products.nc", 
        spco2_unfilled="https://zenodo.org/record/5482547/files/SeaFlux_v2021.04_spco2_SOCOM_unfilled_1982-2019.nc",   
        spco2_filler="https://zenodo.org/record/5482547/files/SeaFlux_v2021.04_spco2_filler_1990-2019.nc",
        pco2atm="https://zenodo.org/record/5482547/files/SeaFlux_v2021.04_pco2atm_1982-2020.nc",    
        area="https://zenodo.org/record/5482547/files/SeaFlux_v2021.04_area_ocean.nc",   
        ice="https://zenodo.org/record/5482547/files/SeaFlux_v2021.04_ice_1982-2020.nc",    
        sol="https://zenodo.org/record/5482547/files/SeaFlux_v2021.04_solWeis74_1982-2020.nc",  
        kw="https://zenodo.org/record/5482547/files/SeaFlux_v2021.04_kw_quadratic_scaled_1982-2020.nc")

    msg = "var_name must be in variables: {}".format(str(list(variables)).replace("'", ""))

    if isinstance(var_name, (list, tuple, np.ndarray)):
        assert all([v in variables.keys() for v in var_name]), msg
        out = [seaflux(v, save_dir=save_dir) for v in var_name]
        return xr.merge(out)
    
    elif isinstance(var_name, str):
        assert var_name in variables.keys(), msg
        url = variables[var_name]
        fname = download_file(url, path=save_dir, progress=True)
        return xr.open_dataset(fname,  chunks={}).conform()