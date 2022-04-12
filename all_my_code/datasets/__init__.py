from functools import cached_property
from . import climate_indicies, regions

class _amc_Data:
    """
    Useful everyday datasets that are quick to download

    Note that data is cached in memory and not downloaded to disk
    
    Includes:
        - mauna loa xco2
        - ocean nino index
        - southern annualar mode
        - pacific decadal oscillation
        - fay and mckinley 2014 CO2 biomes
        - reccap2 regions for global CO2 analysis
    """
    def __init__(self):
        pass

    @cached_property
    def mauna_loa_xco2(self):
        from . climate_indicies import mauna_loa_xco2
        return mauna_loa_xco2()

    @cached_property
    def ocean_nino_index(self):
        from . climate_indicies import ocean_nino_index
        return ocean_nino_index()

    @cached_property
    def pacific_decadal_oscillation(self):
        from . climate_indicies import pacific_decadal_oscillation
        return pacific_decadal_oscillation()

    @cached_property
    def southern_annular_mode(self):
        from . climate_indicies import southern_annular_mode
        return southern_annular_mode()

    @cached_property
    def fay_any_mckinley_2014_biomes(self):
        from . regions import fay_any_mckinley_2014_biomes
        return fay_any_mckinley_2014_biomes()
    
    @cached_property
    def reccap2_regions(self):
        from . regions import reccap2_regions
        return reccap2_regions()