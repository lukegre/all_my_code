def test_fay_mckinley():
    import all_my_code as amc
    from xarray import Dataset

    ds = amc.data.fay_any_mckinley_2014_biomes
    is_dataset = isinstance(ds, Dataset)
    assert is_dataset
    assert hasattr(ds, "mean_biomes")
