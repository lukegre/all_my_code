import all_my_code as amc


def test_lon_360():
    ds = amc.data.fay_any_mckinley_2014_biomes
    func = amc.munging.grid.lon_0E_360E

    lon360 = func(ds).lon

    assert lon360.min() >= 0
    assert lon360.max() > 180


def test_make_like_array():
    from numpy import isclose

    func = amc.munging.grid._make_like_array

    ds1 = func(1.0)
    assert isclose(ds1.shape, [180, 360]).all()

    ds2 = func(2.0)
    assert isclose(ds2.shape, [90, 180]).all()
