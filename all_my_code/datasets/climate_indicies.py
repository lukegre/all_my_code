import pandas as pd
import numpy as np
import xarray as xr


def southern_annular_mode(freq='monthly'):

    if freq == 'monthly':
        da = _southern_annular_mode_monthly()
    elif (freq == 'seasonal'):
        da = _southern_annular_mode_seasonal()
    elif freq == 'annual':
        da = _southern_annular_mode_annual()
    else:
        raise ValueError("freq must be 'monthly' or 'seasonal' or 'annual'")

    da = da.assign_attrs(
        website='https://legacy.bas.ac.uk/met/gjma/sam.html',
        reference=(
            "Marshall, G. J., 2003: Trends in the Southern Annular Mode from "
            "observations and reanalyses. J. Clim., 16, 4134-4143, "
            "doi:10.1175/1520-0442%282003%29016<4134%3ATITSAM>2.0.CO%3B2"),
        description=(
            "The station-based index of the Southern Annular Mode (SAM) is based "
            "on the zonal pressure difference between the latitudes of 40S and 65S. "
            "As such, the SAM index measures a see-saw of atmospheric mass between "
            "the middle and high latitudes of the Southern Hemisphere.  Positive "
            "values of the SAM index correspond with stronger-than-average westerlies "
            "over the mid-high latitudes (50S-70S) and weaker westerlies in the "
            "mid-latitudes (30S-50S).  The SAM is the leading mode of variability "
            "in the SH atmospheric circulation on month-to-month and interannual "
            "timescales. SAM variability has large impacts on Antarctic surface "
            "temperatures, ocean circulation, and many other aspects of SH climate. "
            "The station-based SAM index, which extends back to 1957, uses records "
            "from six stations at roughly 65S and six stations at roughly 40S. "
            "It was developed as an alternative to reanlayses-based indices, which "
            "are of questionable quality before ~1979. The station-based index is "
            "defined on monthly, seasonal, and annual timescales.  The NOAA Climate "
            "Prediction Center computes a daily SAM index based on reanalyses. "
            "Antarctic Oscillation is another name for the SAM. "
            "https://climatedataguide.ucar.edu/climate-data/marshall-southern-annular-mode-sam-index-station-based"
        )
    )

    return da  


def _southern_annular_mode_monthly():
    from all_my_code.munging.date_utils import decimal_year_to_datetime
    url = 'http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.txt'
    df = pd.read_fwf(url, index_col=0, skiprows=1, names=range(12))
    df = df.T.unstack().reset_index()
    df.columns = ['year', 'month', 'southern_annular_mode']

    times = df.year + df.month / 12 + 2 / 365
    df['time'] = decimal_year_to_datetime(times).values.astype('datetime64[M]')
    da = df.set_index('time')['southern_annular_mode'].to_xarray().dropna('time')
    da = da.assign_attrs(source=url)

    return da


def _southern_annular_mode_seasonal():
    from all_my_code.munging.date_utils import decimal_year_to_datetime

    url = 'http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.seas.txt'
    df = pd.read_fwf(url, index_col=0)

    df = (
        df
        .drop(columns=['ANN'])
        .transpose()
        .unstack()
        .reset_index()
        .replace('AUT', 2)
        .replace('WIN', 5)
        .replace('SPR', 8)
        .replace('SUM', 11)
        .rename(columns={'level_0': 'year', 'level_1': 'month', 0: 'southern_annular_mode'}))

    times = df.year + df.month / 12 + 2 / 365
    df['time'] = decimal_year_to_datetime(times).values.astype('datetime64[M]')
    da = df.set_index('time')['southern_annular_mode'].to_xarray().dropna('time')
    da = da.assign_attrs(source=url)

    return da


def _southern_annular_mode_annual():
    from all_my_code.munging.date_utils import decimal_year_to_datetime

    url = 'http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.seas.txt'
    df = pd.read_fwf(url, index_col=0)

    df = df[['ANN']]
    df.columns = ['southern_annular_mode']
    da = df.southern_annular_mode.to_xarray().rename(index='time')
    time = da.time.values.astype('timedelta64[Y]').astype('timedelta64[D]') + np.datetime64('0000-01-05')
    da = da.assign_coords(time=time.astype('datetime64[M]'))
    da = da.assign_attrs(source=url)

    return da


def ocean_nino_index():
    """Download the Ocean Nini Index data from the NOAA website."""

    df = pd.read_fwf(
        "http://www.esrl.noaa.gov/psd/data/correlation/oni.data", 
        infer_nrows=10, 
        index_col=0, 
        names=range(1, 13), 
        skiprows=1,
        na_values=99.9,
    ).iloc[:2022-1950].astype(float).T.unstack(level=0)

    df = df.reset_index()
    df.columns = ['year', 'month', 'oni']
    df = df.astype(dict(year=str, month=str, oni=float))
    df['time'] = pd.to_datetime(df.year + '-' + df.month.str.zfill(2))
    oni = df.set_index('time')['oni'].to_xarray()
    
    dt = np.timedelta64(14, 'D')
    oni = oni.assign_coords(time=lambda x: x.time.values.astype('datetime64[M]') + dt)
    oni = oni.assign_attrs(
        source="http://www.esrl.noaa.gov/psd/data/correlation/oni.data",
        webpage="https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni",
        description=(
            "The Oceanic Niño Index (ONI) is NOAA's primary index for tracking "
            "the ocean part of ENSO, the El Niño-Southern Oscillation climate "
            "pattern. The ONI is the rolling 3-month average temperature "
            "anomaly—difference from average—in the surface waters of the "
            "east-central tropical Pacific (5N-5S, 170W-120W). "
            "To be classified as a full-fledged El Niño or La Niña, the anomalies "
            "must exceed +0.5C or -0.5C for at least five consecutive months. "))

    return oni


def mauna_loa_xco2():
    """Download atmospheric CO2 data from Mauna Loa Observatory."""
    from ..munging.date_utils import decimal_year_to_datetime

    cols = np.array([
        'year',  # 0
        'month', 
        'decimal_year',  # 2
        'monthly_avg', 
        'deseasonalised',  # 4
        'days', 
        'stdev_of_days',  # 6
        'uncertainty_of_mon_avg'])
    
    df = pd.read_fwf(
        "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt", 
        comment='#', 
        skiprows=53, 
        names=cols
    )

    dt = np.timedelta64(14, 'D')
    df['time'] = decimal_year_to_datetime(df.decimal_year).values.astype('datetime64[M]') + dt

    da = df.set_index('time')[cols[3:5]].to_xarray()

    da = da.assign_attrs(
        source="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
        webpage="https://gml.noaa.gov/ccgg/trends/data.html",
        units='ppm',
        citation=(
            "Dr. Pieter Tans, NOAA/GML (gml.noaa.gov/ccgg/trends/) and Dr. Ralph Keeling, "
            "Scripps Institution of Oceanography (scrippsco2.ucsd.edu/)."),
        description=(
            "The Mauna Loa data are being obtained at an altitude of 3400 m in the "
            "northern subtropics, and may not be the same as the globally averaged "
            "CO2 concentration at the surface."
            "Data from March 1958 through April 1974 have been obtained by C. David Keeling "
            "of the Scripps Institution of Oceanography (SIO) and were obtained from the "
            "Scripps website (scrippsco2.ucsd.edu). "
            "Monthly mean CO2 constructed from daily mean values "
            "Scripps data downloaded from http://scrippsco2.ucsd.edu/data/atmospheric_co2 "
            "Monthly values are corrected to center of month based on average seasonal "
            "cycle. Missing days can be asymmetric which would produce a high or low bias. "
            "Missing months have been interpolated, for NOAA data indicated by negative stdev "
            "and uncertainty. "
        ))
    
    return da