def test_graven_fit():
    import all_my_code as amc

    ethz = amc.data.oceansoda_ethz(version=2020)
    pco2 = ethz.spco2
    mask = amc.datasets.masks.make_pco2_seasonal_mask(pco2)

    da = pco2.spatial.aggregate_region(mask)

    sc = da.stats.seascycl_fit_graven()

    print(sc)
