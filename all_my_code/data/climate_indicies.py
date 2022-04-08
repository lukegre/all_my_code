import pandas as pd
import numpy as np
import xarray as xr

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