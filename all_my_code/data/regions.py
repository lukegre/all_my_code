

def fay_any_mckinley_2014_biomes(save_dir="~/Downloads"):
    """
    Download the Fay and McKinley (2014) biomes and conform
    """
    from xarray import open_dataset, merge
    from pandas import DataFrame
    from ..files.download import download_file


    url = "https://epic.awi.de/id/eprint/34786/19/Time_Varying_Biomes.nc"
    fname = download_file(url, path=save_dir)
    fm14 = open_dataset(fname).conform()

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

    fm14 = merge([fm14, names])

    return fm14