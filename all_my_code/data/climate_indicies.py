import pandas as pd
import numpy as np
import xarray as xr

def get_oni_data():
    """Downloads the Ocean Nini Index data from the NOAA website."""

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
    oni = df.set_index('time')['oni'].loc['1982':'2020']

    oni = oni.to_xarray()

    return oni


def manua_loa_xco2():
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

    df = df.set_index('time')[cols[3:]]
    
    df = df.to_xarray()
    
    return df