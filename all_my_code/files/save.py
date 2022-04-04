from pkgutil import get_data
from functools import wraps
import xarray as xr


@xr.register_dataset_accessor("to_netcdf_with_compression")
@xr.register_dataarray_accessor("to_netcdf_with_compression")
class Save(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, sname, overwrite=False, compression='int16', **kwargs):
        """
        Save data to a netCDF file with compression.

        Parameters
        ----------
        sname : str
            Name of the file to save to.
        overwrite : bool [False]
            Overwrite the file if it already exists.
        compression : str [int16]
            The type of compression to use. Options are 
                - `int16` (lossy)
                - `zip` (lossless)
                - `none` (no compression)
        **kwargs : dict
            Additional keyword arguments to pass to `xarray.Dataset.to_netcdf`.
        
        Returns
        -------
        sname : str
            The savename of the saved file
        """
        import os

        ds = self._obj
        
        if isinstance(ds, xr.DataArray):
            try:
                ds = ds.to_dataset()
            except ValueError as E:
                msg = (
                    'DataArray has no name and cannot be converted to xr.Dataset. '
                    'please assign a name with  xr.DataArray.rename(new_name)'
                )
                raise ValueError(msg)


        # function will return an error if the file exists and overwrite is False
        if os.path.exists(sname) and not overwrite:
            print(f'File already exists: {sname}')
            return sname

        if compression == 'int16':
            encoding = get_dataset_compression_encoding(ds)
        elif compression == 'zip':
            encoding = {k: {'complevel': 4, 'zlib': True} for k in ds}
        else:
            encoding = {k: {} for k in ds}
        
        ds.to_netcdf(sname, encoding=encoding, **kwargs)
        return sname
    

def save_dataset_with_compression(ds, sname, max_percentile=99.999):
    """
    Save a dataset with lossy compression. 

    Here we automatically determine the offset and scale factor of the dataset. 
    The min and max values of the dataset are set to the 0.001 and 99.999th 
    percentiles respectively. The data is then scaled based on the range and 
    the maximum size of a 16-bit integer. Further, the file is compressed with 
    zlib to a compression level of 1 (fast speed with minimal difference between
    higher values).

    Parameters
    ----------
    ds: xr.Dataset
        The dataset you want to save
    sname: string
        The destination file name of the dataset
    max_percentile: float
        The upper limit of the data. If you set this to 100, 
        you will use the maximum and minimum values to establish the range. 
        However, you will lose precision by doing this. Recommended to set 
        this to 99.999 to eliminate outliers in your data and gain precision

    Returns
    -------
    None
    """
    encoding = get_dataset_compression_encoding(ds, max_percentile=max_percentile)
    ds.to_netcdf(sname, encoding=encoding)


def get_dataset_compression_encoding(ds, max_percentile=99.999):
    """
    Creates the encoding to compress data. 
    
    WARNING: 
        this loses precision. If your range is large and your 
        precision is important, then do not use this method. 

    Parameters
    ----------
    ds: xr.Dataset
        The input dataset - does not work with data arrays
    max_method:
    """
    encoding = {}
    for k in ds:
        encoding[k] = get_int16_compression_encoding(
            ds[k],
            max_percentile=max_percentile)
    return encoding


def get_int16_compression_encoding(da, max_percentile=99.999):
    return get_int_encoding(da, 16, max_percentile)


def get_int_encoding(da, n=16, max_percentile=100):
    """
    Calculate encoing offset and scale factor for int conversion
    I recommend int16 for a balance between good compression and 
    preservation of precision. 
    
    Parameters
    ----------
    da: xr.DataArray
        the data array you'd like to compress
    n: int
        integer bit number [8, 16, 32]
        recommend 16
    max_percentile: float [100]
        Value to set the maximum limit of the data. 
        I recommend values greater than 99.99
    """
    
    if max_percentile == 100:
        vmin = da.min().values
        vmax = da.max().values
    elif max_percentile < 100:
        quantile = max_percentile / 100
        q = [1 - quantile, quantile] 
        vmin, vmax = da.quantile(q).values
    else:
        raise ValueError('max_percentile must be <= 100')
        
    n_ints = 2**n
    max_scaled = n_ints / 2 - 1
    min_scaled = 1 - n_ints / 2
    scaling_range = max_scaled - min_scaled
    
    # stretch/compress data to the available packed range
    scale_factor = (vmax - vmin) / scaling_range

    # translate the range to be symmetric about zero
    add_offset = vmin + max_scaled * scale_factor
    
    # fill_value is 1 less than the minimum scaled value
    fill_value = min_scaled - 1
    
    encoding = dict(
        _FillValue=fill_value,
        add_offset=add_offset,
        scale_factor=scale_factor,
        complevel=1,
        zlib=True, 
        dtype=f'int{n}',
    )

    return encoding