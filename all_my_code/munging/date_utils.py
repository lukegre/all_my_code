def datetime64ns_to_lower_order_datetime(arr):
    """
    If monthly data is saved as datetime64[ns], will return datetime64[M]

    This is useful if you want to calculate the diff in the appropriate
    units
    """
    from numpy import array, isnat

    # test if dates
    arr = array(arr)
    if isnat(arr).any():
        return arr

    # get the unit of the time array
    unit = get_time_step_unit(arr)

    # convert time units to appropriate units
    # dtype: datetime64 must be attached to unit
    # must be float when astype(float) is applied
    time_converted = arr.astype(f"datetime64[{unit}]")

    return time_converted


def get_time_step_unit(time_arr):
    from numpy import array, diff, NaN, nanmedian
    from pandas import Series

    time_ns = array(time_arr).astype("datetime64[ns]").astype(float)
    # get the difference between time steps
    delta_time = diff(time_ns)

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

    return unit


def decimal_year_to_datetime(decimal_year):
    """
    Convert a decimal year to a pandas timestamp.
    """
    from collections.abc import Iterable
    from pandas import DatetimeIndex, to_datetime

    if isinstance(decimal_year, Iterable):
        return DatetimeIndex([decimal_year_to_datetime(y) for y in decimal_year])

    from calendar import isleap

    def get_int_and_frac(number, frac_scaler=1):
        intiger = int(number)
        fraction = number - intiger
        fraction *= frac_scaler
        return intiger, fraction

    # convert a decimal year to a pandas timestamp
    year, ddays = get_int_and_frac(
        decimal_year, 366 if isleap(int(decimal_year)) else 365
    )
    ddays += 1
    days, dhours = get_int_and_frac(ddays, 24)
    hour, dmins = get_int_and_frac(dhours, 60)
    mins, dsecs = get_int_and_frac(dmins, 60)
    secs, ms = get_int_and_frac(dsecs)
    timestamp = to_datetime(
        f"{year:d}-{days:d} {hour:d}:{mins:d}:{secs:d}", format="%Y-%j %H:%M:%S"
    )

    return timestamp


def datestring_to_datetime(string, return_date=False):
    """
    Will try to convert a datestring into a datetime if it
    matches conventional date formats. Supports years
    1870 to 2129. May confuse American (mm-dd-yyyy) format
    if the day < 12.
    """
    import re
    from pandas import to_datetime

    if isinstance(string, list):
        return [datestring_to_datetime(s) for s in string]

    year = "(?P<year>[12][0189][012789][0-9])"
    mon = "(?P<mon>[01][0-9])"
    day = "(?P<day>[0-3][0-9])"
    sep = "(?P<sep>.?)"

    patterns = [
        "{year}",
        "{year}{sep}{mon}",
        "{mon}{sep}{day}.?{year}",
        "{day}{sep}{mon}.?{year}",
        "{year}{sep}{mon}.?{day}",
    ]

    out = 0
    best_pattern = None
    for pattern in patterns:
        p = pattern.format(year=year, mon=mon, day=day, sep=sep)
        matches = re.match(p, string)
        if matches is None:
            continue
        groups = matches.groupdict()
        if len(groups) > 0:
            s = "" if "sep" not in groups else groups["sep"]
            best_pattern = pattern.format(
                year="%Y",
                mon="%m",
                day="%d",
                sep=s,
            ).replace(".?", s)
            out += 1

    try:
        date = to_datetime(string, format=best_pattern)
        return date.to_numpy()
    except ValueError:
        return string
