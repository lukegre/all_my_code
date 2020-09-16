def convert_time_to_most_suitable_unit(arr):
    from pandas import Series
    from numpy import array, isnat, diff, NaN, nanmedian

    # test if dates
    arr = array(arr)
    if isnat(arr).any():
        return arr

    # convert to datetime[ns] floats
    time = array(arr).astype("datetime64[ns]").astype(float)

    # get the difference between time steps
    delta_time = diff(time)

    # approximate the best unit (without losing info)
    time_denominators = dict(ns=1, s=1e9, m=60, h=60, D=24, M=30, Y=12)

    dt_as_frac_of_unit = Series(index=time_denominators.keys())
    denominator = 1
    for key in time_denominators:
        denominator *= time_denominators[key]
        frac = nanmedian(delta_time / denominator)
        # only units that will not lose time are kept
        dt_as_frac_of_unit[key] = frac if frac >= 1 else NaN

    # if the difference is not near enough the unit, exclude it
    # e.g. 35 day interval will eliminate Month as a unit
    if not ((dt_as_frac_of_unit - 1) < 0.05).any():
        dt_as_frac_of_unit = dt_as_frac_of_unit.where(lambda a: (a - 1) >= 1)
    unit = dt_as_frac_of_unit.idxmin()

    # convert time units to appropriate units
    # dtype: datetime64 must be attached to unit
    # must be float when astype(float) is applied
    time_converted = arr.astype(f"datetime64[{unit}]")

    return time_converted
