

def fay_any_mckinley_2014_biomes():
    """
    Download the Fay and McKinley (2014) biomes and conform

    Data is not downloaded to disk, but loaded to memory. Save as a netcdf file
    to save the data to disk
    """
    from xarray import open_dataset, merge
    from pandas import DataFrame
    from fsspec import open
    import numpy as np


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

    fm14 = merge([fm14, names]).fillna(0)
    for key in fm14:
        da = fm14[key].load()
        if da.dtype == np.float_:
            da = da.astype(np.int8)
            fm14[key] = da

    return fm14


def reccap2_regions():
    import xarray as xr
    import fsspec
    url = (
        "https://github.com/RECCAP2-ocean/R2-shared-resources/raw"
        "/master/data/regions/RECCAP2_region_masks_all_v20210412.nc")
    ds = xr.open_dataset(fsspec.open(url).open())

    ds = ds.conform(
        correct_coord_names=False, 
        drop_0d_coords=False, 
        transpose_dims=False)

    return ds


def seamask():
    return reccap2_regions()['seamask']