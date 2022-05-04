from tempfile import gettempdir
from ..munging.grid import _make_like_array


def fay_any_mckinley_2014_biomes(resolution=1, save_dir=gettempdir()):
    """
    Download the Fay and McKinley (2014) biomes and conform

    Data is not downloaded to disk, but loaded to memory. Save as a netcdf file
    to save the data to disk
    """
    from xarray import open_dataset, merge
    from pandas import DataFrame
    from ..files.download import download_file
    import numpy as np

    url = "https://epic.awi.de/id/eprint/34786/19/Time_Varying_Biomes.nc"

    fname = download_file(url, path=save_dir, progress=True)
    fm14 = open_dataset(fname).conform(rename_vars_snake_case=True)

    fm14 = fm14.assign_attrs(
        source=url,
        doi="https://essd.copernicus.org/articles/6/273/2014/",
    )

    names = (
        DataFrame(
            {
                1: ["North Pacific Ice", "NP_ICE"],
                2: ["North Pacific Subpolar Seasonally Stratified", "NP_SPSS"],
                3: ["North Pacific Subtropical Seasonally Stratified ", "NP_STSS"],
                4: ["North Pacific Subtropical Permanently Stratified", "NP_STPS"],
                5: ["West pacific Equatorial", "PEQU_W"],
                6: ["East Pacific Equatorial", "PEQU_E"],
                7: ["South Pacific Subtropical Permanently Stratified", "SP_STPS"],
                8: ["North Atlantic Ice", "NA_ICE"],
                9: ["North Atlantic Subpolar Seasonally Stratified", "NA_SPSS"],
                10: ["North Atlantic Subtropical Seasonally Stratified", "NA_STSS"],
                11: ["North Atlantic Subtropical Permanently Stratified", "NA_STPS"],
                12: ["Atlantic Equatorial", "AEQU"],
                13: ["South Atlantic Subtropical Permanently Stratified", "SA_STPS"],
                14: ["Indian Ocean Subtropical Permanently Stratified", "IND_STPS"],
                15: ["Southern Ocean Subtropical Seasonally Stratified", "SO_STSS"],
                16: ["Southern Ocean Subpolar Seasonally Stratified", "SO_SPSS"],
                17: ["Southern Ocean Ice", "SO_ICE"],
            }
        )
        .transpose()
        .rename(columns={0: "biome_name", 1: "biome_abbrev"})
        .to_xarray()
        .rename(index="biome_number")
    )

    ds = merge([fm14, names]).fillna(0)
    for key in ds:
        da = ds[key].load()
        if da.dtype == np.float_:
            da = da.astype(np.int8)
            ds[key] = da

    if resolution == 1:
        return ds
    else:
        like = _make_like_array(resolution)
        ds = ds.reindex_like(like, method="nearest")
        return ds


def reccap2_regions(resolution=1, save_dir=gettempdir()):
    import xarray as xr
    from ..files.download import download_file

    url = (
        "https://github.com/RECCAP2-ocean/R2-shared-resources/raw"
        "/master/data/regions/RECCAP2_region_masks_all_v20210412.nc"
    )
    fname = download_file(url, path=save_dir, progress=True)
    ds = xr.open_dataset(fname)
    ds = ds.conform(
        correct_coord_names=False, drop_0d_coords=False, transpose_dims=False
    )

    if resolution == 1:
        return ds
    else:
        like = _make_like_array(resolution)
        ds = ds.reindex_like(like, method="nearest")
        return ds


def seafrac(resolution=1, save_dir=gettempdir()):
    """
    Returns a mask that shows the fraction of ocean based on ETOPO1 data

    Parameters
    ----------
    resolution : float
        Resolution of the output resolution in degrees

    Returns
    -------
    xarray.Dataset
        Dataset containing the fraction of ocean
    """
    return make_etopo_mask(res=resolution, save_dir=save_dir).sea_frac


def seamask(resolution=1, save_dir=gettempdir()):
    """
    Returns a mask for ocean pixels where sea fraction > 50%

    Based on the ETOPO1 data scaled to the appropriate resolution
    Note that this might result in regions that are below the geoid
    0 level being marked as ocean - for example, this is the case for
    some of the Netherlands.

    Parameters
    ----------
    resolution : float
        Resolution of the output resolution in degrees

    Returns
    -------
    xarray.Dataset
        a boolean mask of where sea fraction > 50%
    """
    from xarray import set_options

    seafraction = seafrac(resolution=resolution, save_dir=save_dir)
    with set_options(keep_attrs=True):
        seamask = (seafraction > 0.5).assign_attrs(
            description="sea mask where sea fraction > 50%"
        )
    return seamask


def ar6_regions(resolution=1, subset="all"):
    """
    Fetches AR6 regions using the regionmask package

    Parameters
    ----------
    resolution : float
        Resolution of the output resolution in degrees
    subset : str ['all', 'land', 'ocean']
        Subset of the AR6 regions to return.

    Returns
    -------
    xarray.Dataset
        Dataset containing the AR6 regions with two variables:
            - 'region' : the region number
            - 'names' : the name of the region
    """
    from all_my_code.munging.grid import _make_like_array
    import regionmask
    from xarray import DataArray

    ar6 = regionmask.defined_regions.ar6
    regions = getattr(ar6, subset)

    like = _make_like_array(resolution)

    mask = regions.mask(like.lon, like.lat).astype("int8").to_dataset(name="region")

    mask["names"] = DataArray(
        regions.names,
        dims=("number",),
        coords={"number": regions.numbers},
        attrs={"source": "Python package <regionmask>"},
    )
    return mask


def topography(resolution=1, save_dir=gettempdir()):
    """
    Get ETOPO1 topography data resampled to the desired resolution

    Parameters
    ----------
    resolution : float
        Resolution of the output resolution in degrees

    Returns
    -------
    xarray.Dataset
        Dataset containing the topography average
        deviation based on 1 arc-minute resolution
    """
    return make_etopo_mask(res=resolution, save_dir=save_dir).topo_avg


def make_etopo_mask(res=1, save_dir=gettempdir(), delete_intermediate_files=True):
    """
    Downloads ETOPO1 data and creates a new file containing
    - average height
    - standard deviation of height (from pixels in original file)
    - fraction of ocean
    """
    import xarray as xr

    ds = _fetch_etopo(
        save_dir=save_dir, delete_intermediate_files=delete_intermediate_files
    )

    # the coarsening step is determined by 60min / res to get to degrees
    d = int(60 * res)
    out = xr.Dataset()
    coarse = ds.z.coarsen(lat=d, lon=d)
    out["topo_avg"] = coarse.mean(keep_attrs=True)
    out["topo_std"] = coarse.std(keep_attrs=True)
    out["sea_frac"] = (ds.z < 0).coarsen(lat=d, lon=d).sum() / d**2
    out = out.assign_attrs(
        **ds.attrs,
        description="Topography data from ETOPO1 resampled to the given resolution.",
    )

    return out


def _fetch_etopo(save_dir=gettempdir(), delete_intermediate_files=True):
    """
    Fetch ETOPO1 data and return it as an xarray.Dataset

    The ETOPO1 data will be downloaded if it does not exist.

    Parameters
    ----------
    save_dir : str
        Path to the directory where the data should be stored. Default
        is the system's temporary directory.
    delete_intermediate_files : bool [False]
        If True, delete the intermediate files after the data is downloaded

    Returns
    -------
    xarray.Dataset
        Dataset containing the ETOPO1 data that has been conformed
    """
    import xarray as xr
    from ..files.download import download_file
    from pathlib import Path as posixpath
    import os

    url = (
        "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data"
        "/ice_surface/cell_registered/netcdf/ETOPO1_Ice_c_gmt4.grd.gz"
    )

    fname = os.path.join(os.path.expanduser(save_dir), posixpath(url).name).replace(
        "grd.gz", "nc"
    )
    if not os.path.isfile(fname):
        print(
            "Downloading ETOPO1 data - this is only done once (unless you delete the "
            f"tmp directory). The final file will be saved at {fname} and "
            "has a size of roughly 250MB. All other intermediate files will be deleted."
        )
        tmp_fname = download_file(url, path=save_dir)
        tmp_data = xr.open_dataset(tmp_fname).assign_attrs(source=url, dest=fname)
        tmp_data.astype("int16").to_netcdf(
            fname, encoding={"z": {"complevel": 1, "zlib": True}}
        )
        if delete_intermediate_files:
            os.remove(tmp_fname)

    # load with mfdataset to ensure that we're chunking the data
    # then we conform the data so that x, y are renamed to lat, lon
    ds = xr.open_mfdataset([fname]).conform()
    return ds


def hemisphere_sign(resolution=1):
    """
    Returns a mask where the NH is -1 and SH is +1

    Parameters
    ----------
    resolution : float
        Resolution of the output resolution in degrees

    Returns
    -------
    xarray.DataArray
        DataArray containing the hemisphere sign
    """
    blank = _make_like_array(resolution)
    hem_flip = blank.fillna(1).where(lambda x: x.lat > 0).fillna(-1)
    return hem_flip


def make_pco2_seasonal_mask(pco2, res=1, eq_lat=10, high_lat=65):
    """
    Make a mask for the given pCO2 seasonal cycle.

    Parameters
    ----------
    pco2 : xr.DataArray
        pCO2 seasonal from which the seasonal cycle will be calculated
        and the mask will be constructed.
    res : int
        Resolution of the mask.
    eq_lat : float
        Latitude for the Equatorial boundary
    high_lat : float
        Latitude for the High latitude boundary

    Returns
    -------
    mask : xr.DataArray
        Boolean mask for the given pCO2 seasonal cycle.
    """
    from xarray import concat

    mask1 = _make_zonal_mask(1, 20, high_lat)
    mask2 = _make_zonal_mask(1, eq_lat, 50)

    hem_flip = hemisphere_sign(res)

    seascyl = pco2.stats.seascycl_fit_climatology().jja_minus_djf
    jfm = seascyl.sel(month=[12, 1, 2]).mean("month")  # JFM
    jas = seascyl.sel(month=[6, 7, 8]).mean("month")  # JAS
    diff = (jfm - jas) * hem_flip

    high_lats = diff.where(mask1) > 0
    low_lats = mask2 & ~high_lats & diff.notnull()

    nh_hl = high_lats.where(lambda x: (x > 0) & (hem_flip > 0)) * 1
    nh_ll = low_lats.where(lambda x: (x > 0) & (hem_flip > 0)) * 2
    sh_ll = low_lats.where(lambda x: (x > 0) & (hem_flip < 0)) * 3
    sh_hl = high_lats.where(lambda x: (x > 0) & (hem_flip < 0)) * 4

    seas_cycle_mask = concat([nh_hl, nh_ll, sh_ll, sh_hl], "tmp").sum("tmp")

    return seas_cycle_mask


def _make_zonal_mask(res, min_lat, max_lat):
    """
    Make a mask for the given latitude range.

    The mask will make two bands - one for the northern
    hemisphere and one for the southern hemisphere.

    Parameters
    ----------
    res : int
        Resolution of the mask.
    min_lat : float
        Minimum latitude for the mask (equator)
    max_lat : float
        Maximum latitude for the mask (pole)

    Returns
    -------
    mask : xr.DataArray
        Boolean mask for the given latitude range.
    """

    min_lat = abs(min_lat)
    max_lat = abs(max_lat)

    blank = _make_like_array(res).fillna(1)
    y = blank.lat

    mask = blank.where(
        ((y >= -max_lat) & (y <= -min_lat)) | ((y >= min_lat) & (y <= max_lat))
    )

    mask = mask.fillna(0).astype(bool)

    return mask


def basins_sans_southern_ocean(resolution=1, save_dir=gettempdir()):
    """
    Make a basin mask that includes Atlantic, Pacific, Indian,
    and Arctic basins. The first three extend southward all the
    way to the Antarctic.

    Parameters
    ----------
    resolution : float
        Resolution of the output resolution in degrees
    save_dir : str
        Path to the directory where the data should be stored. Default
        is the system's temporary directory.

    Returns
    -------
    xarray.DataArray
        Intiger mask ranging from 0 - 4 with the following basins:
        0 = land / marginal seas
        1 = Atlantic (-70 : 20)
        2 = Pacific (145 : -70)
        3 = Indian (20 : 145)
        4 = Arctic (as defined in RECCAP2)

    Note
    ----
    Uses the reccap2 open_ocean mask as a base mask
    """
    mask = reccap2_regions.open_ocean(resolution=resolution, save_dir=save_dir)

    atlantic = (mask & ((mask.lon > -70) & (mask.lon < 20))) | (mask == 1)
    pacific = ((mask & ((mask.lon < -70) | (mask.lon > 145))) | (mask == 2)) & (
        mask != 1
    )
    indian = mask & ((mask.lon > 20) & (mask.lon < 145)) & (~mask.isin([1, 2]))
    arctic = mask == 4

    mask_sans_so = (atlantic * 1) + (pacific * 2) + (indian * 3) + (arctic * 4)
    return mask_sans_so
