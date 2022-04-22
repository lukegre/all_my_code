from functools import cached_property as _cached_property, wraps as _wraps

from . import climate_indicies, masks, carbon, other


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
        self.res = 0.25
        self.sam_freq = "monthly"
        self.download_dest = "~/Data/cached"

    def __repr__(self):
        def make_pretty_dataset_repr(funclist):
            string = ""
            for func in funclist:
                loaded = "*" if func in self.__dict__ else ""
                if loaded:
                    da = getattr(self, func)
                    dims = []
                    for d in da.dims:
                        dims += (f"{d}: {da[d].size}",)
                    dims = "[" + ", ".join(dims) + "]"
                else:
                    dims = ""
                string += f"\n{loaded: >4} {func} {dims}"
            return string

        string = f"<all_my_code.data(resolution={self.res}, "
        string += f"southern_annular_mode_freq='{self.sam_freq}', "
        string += f"download_dest='{self.download_dest}')>"
        string += "\nContains the following datasets (* = cached in memory):"

        funcnames = [m for m in dir(self) if not m.startswith("_")]
        for drop in ["res", "sam_freq", "set_defaults"]:
            funcnames.remove(drop)

        regions = [f for f in funcnames if hasattr(masks, f)]
        indices = [f for f in funcnames if hasattr(climate_indicies, f)]
        co2data = [f for f in funcnames if hasattr(carbon, f)]
        string += "\n  MASKS"
        string += make_pretty_dataset_repr(regions)
        string += "\n  CLIMATE INDICES"
        string += make_pretty_dataset_repr(indices)
        string += f"\n  CARBON DATASETS [{self.download_dest}]"
        string += make_pretty_dataset_repr(co2data)
        string += f"\n  Default CARBON download path can be set with `amc.data.download_dest = <new_path>`"
        string += (
            f"\n  Note that all CARBON data will have to be downloaded at least once"
        )

        return string

    def set_defaults(
        self, resolution=None, southern_annular_mode_freq=None, download_dest=None
    ):
        if southern_annular_mode_freq is not None:
            self.sam_freq = southern_annular_mode_freq
            self.__dict__.pop("southern_annular_mode", None)
        if resolution is not None:
            self.res = resolution
            for key in [
                "seafrac",
                "topography",
                "fay_any_mckinley_2014_biomes",
                "reccap2_regions",
                "ar6_regions",
            ]:
                self.__dict__.pop(key, None)
        if download_dest is not None:
            self.download_dest = download_dest

    @_cached_property
    def mauna_loa_xco2(self):
        from .climate_indicies import mauna_loa_xco2 as func

        return func()

    @_cached_property
    def ocean_nino_index(self):
        from .climate_indicies import ocean_nino_index as func

        return func()

    @_cached_property
    def pacific_decadal_oscillation(self):
        from .climate_indicies import pacific_decadal_oscillation as func

        return func()

    @_cached_property
    def southern_annular_mode(self):
        from .climate_indicies import southern_annular_mode as func

        return func(freq=self.sam_freq)

    @_cached_property
    def sea_ice_area(self):
        from .climate_indicies import sea_ice_area as func

        return func(time="monthly")

    @_cached_property
    def fay_any_mckinley_2014_biomes(self):
        from .masks import fay_any_mckinley_2014_biomes as func

        return func(resolution=self.res).load()

    @_cached_property
    def reccap2_regions(self):
        from .masks import reccap2_regions as func

        return func(resolution=self.res).load()

    @_cached_property
    def seafrac(self):
        from .masks import seafrac as func

        return func(resolution=self.res, save_dir=self.download_dest)

    @_cached_property
    def seamask(self):
        from .masks import seamask as func

        return func(resolution=self.res, save_dir=self.download_dest)

    @_cached_property
    def ar6_regions(self):
        from .masks import ar6_regions as func

        return func(resolution=self.res)

    @_cached_property
    def topography(self):
        from .masks import topography as func

        return func(resolution=self.res, save_dir=self.download_dest)

    @_wraps(carbon.oceansoda_ethz)
    def oceansoda_ethz(self, *args, **kwargs):
        from .carbon import oceansoda_ethz as func

        props = dict(save_dir=self.download_dest)
        props.update(kwargs)
        return func(*args, **props)

    @_wraps(carbon.socat_gridded)
    def socat_gridded(self, *args, **kwargs):
        from .carbon import socat_gridded as func

        props = dict(save_dir=self.download_dest)
        props.update(kwargs)
        return func(*args, **props)

    @_wraps(carbon.seaflux)
    def seaflux(self, *args, **kwargs):
        from .carbon import seaflux as func

        props = dict(save_dir=self.download_dest)
        props.update(kwargs)
        return func(*args, **props)

    @_wraps(other.get_woa18)
    def woa18(self, *args, **kwargs):
        from .other import get_woa18 as func

        props = dict(save_dir=self.download_dest)
        props.update(kwargs)
        return func(*args, **props)
