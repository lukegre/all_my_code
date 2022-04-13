from tempfile import gettempdir


def fay_any_mckinley_2014_biomes(resolution=1):
    """
    Download the Fay and McKinley (2014) biomes and conform

    Data is not downloaded to disk, but loaded to memory. Save as a netcdf file
    to save the data to disk
    """
    from xarray import open_dataset, merge
    from pandas import DataFrame
    from fsspec import open
    import numpy as np
    from ..munging import _make_like_array

    url = "https://epic.awi.de/id/eprint/34786/19/Time_Varying_Biomes.nc"
    file_obj = open(url).open()
    fm14 = open_dataset(file_obj).conform(rename_vars_snake_case=True)

    fm14 = fm14.assign_attrs(
        source=url,
        doi="https://essd.copernicus.org/articles/6/273/2014/",
    )

    names = (
        DataFrame({
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
            17: ["Southern Ocean Ice", "SO_ICE"],})
        .transpose()
        .rename(columns={0:'biome_name', 1:'biome_abbrev'})
        .to_xarray()
        .rename(index='biome_number')
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
        ds = ds.reindex_like(like, method='nearest')
        return ds


def reccap2_regions(resolution=1):
    import xarray as xr
    import fsspec
    from ..munging import _make_like_array
    
    url = (
        "https://github.com/RECCAP2-ocean/R2-shared-resources/raw"
        "/master/data/regions/RECCAP2_region_masks_all_v20210412.nc")
    ds = xr.open_dataset(fsspec.open(url).open())

    ds = ds.conform(
        correct_coord_names=False, 
        drop_0d_coords=False, 
        transpose_dims=False)

    if resolution == 1:
        return ds
    else:
        like = _make_like_array(resolution)
        ds = ds.reindex_like(like, method='nearest')
        return ds


def seafrac(resolution=1/4, save_dir=gettempdir()):
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


def topography(resolution=1/4, save_dir=gettempdir()):
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


def make_etopo_mask(res=1/4, save_dir=gettempdir(), delete_intermediate_files=True):
    """
    Downloads ETOPO1 data and creates a new file containing
    - average height
    - standard deviation of height (from pixels in original file)
    - fraction of ocean 
    """
    import xarray as xr

    ds = _fetch_etopo(save_dir=save_dir, delete_intermediate_files=delete_intermediate_files)

    # the coarsening step is determined by 60min / res to get to degrees
    d = int(60 * res)
    out = xr.Dataset()
    coarse = ds.z.coarsen(lat=d, lon=d)
    out['topo_avg'] = coarse.mean(keep_attrs=True)
    out['topo_std'] = coarse.std(keep_attrs=True)
    out['sea_frac'] = (ds.z < 0).coarsen(lat=d, lon=d).sum() / d**2
    out = out.assign_attrs(
        **ds.attrs,
        description="Topography data from ETOPO1 resampled to the given resolution.")

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
        "/ice_surface/cell_registered/netcdf/ETOPO1_Ice_c_gmt4.grd.gz")
    
    fname = os.path.join(save_dir, posixpath(url).name).replace('grd.gz', 'nc')
    if not os.path.isfile(fname):
        print(
            'Downloading ETOPO1 data - this is only done once (unless you delete the '
            f'tmp directory). The final file will be saved at {fname} '
            'and has a size of roughly 250MB. All other intermediate files will be deleted.'
        )
        tmp_fname = download_file(url, path=save_dir)
        tmp_data = xr.open_dataset(tmp_fname).assign_attrs(source=url, dest=fname)
        tmp_data.astype('int16').to_netcdf(fname, encoding={'z': {'complevel': 1, 'zlib': True}})
        if delete_intermediate_files:
            os.remove(tmp_fname)
            os.remove(tmp_fname + '.gz')
    
    # load with mfdataset to ensure that we're chunking the data
    # then we conform the data so that x, y are renamed to lat, lon
    ds = xr.open_mfdataset([fname]).conform()
    return ds
