from ..files.download import download_file
import xarray as xr
import os


def socat_gridded(
    version="2022", save_dir="~/Data/cached/", delete_intermediate_files=True
):
    from pathlib import Path as posixpath

    url = (
        f"https://www.socat.info/socat_files/v{version}"
        f"/SOCATv{version}_tracks_gridded_monthly.nc.zip"
    )
    fname = posixpath(url).name
    zpath = os.path.join(os.path.expanduser(save_dir), fname)
    fpath = os.path.join(os.path.expanduser(save_dir), fname.replace(".zip", ""))

    if not os.path.exists(fpath):
        dpath = download_file(url, save_dir, verbosity=2)
        ds = xr.open_dataset(dpath).drop("tmnth_bnds")
        ds.to_netcdf_with_compression(fpath)

        if delete_intermediate_files:
            os.remove(dpath)
            os.remove(zpath)
            posixpath(dpath).parent.rmdir()

    return xr.open_dataset(fpath, chunks={}).conform()


def _download_ethz_data(save_dir="~/Data/cached/", version="2021"):
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
    if str(version) == "2020":
        fname = "OceanSODA-ETHZ_1985-2019_v2020b.nc"
        url += f"2.2/data/0-data/{fname}"
    elif str(version) == "2021":
        fname = "OceanSODA-ETHZ_GRaCER_v2021a_1982-2020.nc"
        url += f"3.3/data/0-data/{fname}"
    elif str(version) == "2022":
        fname = "OceanSODA-ETHZ_GRaCER_v2022gcb_1982-2021.nc"
        url = "https://figshare.com/ndownloader/files/36723312?private_link=89374797e3474264282a"  # noqa

    name = retrieve(
        url, None, fname, path=save_dir, downloader=HTTPDownloader(progressbar=True)
    )

    return name


def oceansoda_ethz(save_dir="~/Data/cached/", version="2022", salt_norm=34.5):
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
    ds = xr.open_mfdataset(fname, parallel=True)

    # unifying 2020 to 2021 naming (2020 names are not same as 2021)
    if str(version) == "2020":
        unified_names = dict(
            DIC="dic",
            TA="alk",
            pCO2="spco2",
            pH="ph_total",
            HCO3="hco3",
            CO3="co3",
            omegaAR="omega_ar",
            omegaCA="omega_ca",
            TAstd="talk_uncert",
            pCO2std="spco2_uncert",
        )
    elif str(version) >= "2021":
        unified_names = dict(talk="alk", talk_uncert="alk_uncert")

    ds = ds.rename(unified_names)

    if str(version) <= "2021":
        print("[H+], sDIC and sALK calculated (normed to local long-term mean)")
        ds["h"] = (10 ** (-ds.ph_total) * 1e9).assign_attrs(
            units="nmol/kg",
            description=(
                "note that H+ is calculated from pH total and is thus not " "pure H+."
            ),
        )
    else:
        ds["h"] = (10 ** (-ds.ph_free) * 1e9).assign_attrs(
            units="nmol/kg", description=("H+ is calculated from pH free")
        )

    if isinstance(salt_norm, str):
        if salt_norm == "time":
            s0_norm = ds.salinity.mean("time")
        else:
            raise ValueError(f"salt_norm must be 'time' or a float, not {salt_norm}")
    elif isinstance(salt_norm, (float, int)):
        s0_norm = salt_norm
    ds["sdic"] = ds.dic / ds.salinity * s0_norm
    ds["salk"] = ds.alk / ds.salinity * s0_norm

    return ds


def seaflux(var_name="pco2atm", save_dir="~/Data/cached/"):
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

    base = "https://zenodo.org/record/5482547/files"
    variables = dict(
        spco2_unfilled=f"{base}/SeaFlux_v2021.04_spco2_SOCOM_unfilled_1982-2019.nc",
        spco2_filler=f"{base}/SeaFlux_v2021.04_spco2_filler_1990-2019.nc",
        pco2atm=f"{base}/SeaFlux_v2021.04_pco2atm_1982-2020.nc",
        fgco2=f"{base}/SeaFlux_v2021.04_fgco2_all_winds_products.nc",
        area=f"{base}/SeaFlux_v2021.04_area_ocean.nc",
        ice=f"{base}/SeaFlux_v2021.04_ice_1982-2020.nc",
        sol=f"{base}/SeaFlux_v2021.04_solWeis74_1982-2020.nc",
        kw=f"{base}/SeaFlux_v2021.04_kw_quadratic_scaled_1982-2020.nc",
    )

    msg = "var_name must be in variables: {}".format(
        str(list(variables)).replace("'", "")
    )

    if isinstance(var_name, (list, tuple, np.ndarray)):
        assert all([v in variables.keys() for v in var_name]), msg
        out = [seaflux(var_name=v, save_dir=save_dir) for v in var_name]
        return xr.merge(out)

    elif isinstance(var_name, str):
        assert var_name in variables.keys(), msg
        url = variables[var_name]
        fname = download_file(url, path=save_dir, verbosity=2)
        return xr.open_dataset(fname, chunks={}).conform()


def flux_weighting(wind=["CCMP2", "ERA5", "JRA55"], save_dir="~/Data/cached/"):
    """
    A wrapper around data.seaflux that calculates the flux weighting for
    fast calculation of the potential flux or to weight by flux

    Parameters
    ----------
    wind: list[str] | None
        The wind products to calculate the flux weighting for. Defaults
        to the standard CCMP2, ERA5 and JRA55 trio from the SeaFlux dataset.
        NCEP1 and NCEP2 are also available.
    save_dir: path string
        A path to where the data will be downloaded. Will be expanded so ~
        and relative paths can be used. Defaults to ~/Data/cached/. This
        is passed to data.seaflux, so data is not doubled if already
        downloaded with data.seaflux.

    Returns
    -------
    xr.Dataset:
        The flux weighting dataarray in molC/m2/yr/uatm. The dataset is
        returned as a lazy array, meaning that it is fast if the data is
        already downloaded. But the computation will still need to be
        performed. Multiplying (pCO2sea - pCO2air) * flux_weighting will
        return the flux in molC/m2/yr.
    """

    vars = ["kw", "sol", "ice"]
    sf = seaflux(var_name=vars, save_dir=save_dir).drop("alpha")

    if wind is not None:
        sf = sf.sel(wind=wind)
    sf = sf.mean(["wind"])

    # doing calculations
    ice_free = (1 - sf.ice).fillna(0)  # ice-free ocean
    kw = sf.kw * 87.6  # cm/hr --> m/yr
    sol = sf.sol  # mol/m3/uatm

    flux_weight = (sol * kw * ice_free).assign_attrs(
        product="SeaFlux",
        source="https://zenodo.org/record/5482547#.YmK1L_NBy3I",
        citation="https://doi.org/10.5194/essd-13-4693-2021",
        long_name="flux_weighting",
        description=(
            "can be used to weight pCO2 or multiplied by "
            "(pCO2sea - pCO2atm) to get the air-sea CO2 flux."
        ),
        units="molC/m2/yr/uatm",
        kw_winds=wind,
        kw_parameterisation="quadratic scaled for each wind product",
    )

    return flux_weight
