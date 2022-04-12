import xarray as xr
from functools import wraps as _wraps
from ..utils import add_docs_line1_to_attribute_history, get_unwrapped


def lon_180W_180E(ds, lon_name='lon'):
    """
    Regrid the data to [-180 : 180] from [0 : 360]
    """
    from numpy import isclose

    lon = ds[lon_name].values
    lon180 = (lon - 180) % 360 - 180
    if isclose(lon, lon180).all():
        return ds
    return ds.assign_coords(**{lon_name: lon180}).sortby(lon_name)


def lon_0E_360E(ds, lon_name='lon'):
    """
    Regrid the data to [0 : 360] from [-180 : 180] 
    """
    from numpy import isclose
    
    lon = ds[lon_name].values
    lon360 = lon % 360
    if isclose(lon, lon360).all():
        return ds
    ds = ds.assign_coords(**{lon_name: lon360}).sortby(lon_name)
    return ds
    
      
def coord_05_offset(ds, center=0.5, coord_name='lon'):
    """
    Interpolate data to grid centers.
    Only works for 1deg data
    
    Parameters
    ----------
    ds: xr.Dataset
        the dataset with a coordinate variable variable
    center: float
        the desired center point of the grid points between 0 - 1
    coord_name: str [lon]
        the name of the coordinate 
        
    Returns
    -------
    xr.Dataset: interpolated onto the new grid with the new
        coord being the old coord + center
    """

    def has_coords(ds, checklist=['time', 'lat', 'lon']):
        """
        Check that data has coordinates
        """
        matches = {key: (key in ds.coords) for key in checklist}
        if all(matches.values()):
            return 1
        else:
            return 0

    center = center - (center // 1)
    if has_coords(ds):
        coord = ds[coord_name].values
        mod = coord - (coord // 1)
        # use the modulus to determine if grid centers are correct
        if any(mod != center):
            ds = ds.interp({coord_name: coord + center})
            ds = ds.sel(lat=slice(-90, 90))
            
    return ds


def interp(ds, lon_name='lon', roll_by=10, **kwargs):
    """
    Interpolate and fill the longitude gap in a dataset

    Parameters
    ----------
    ds: xr.Dataset
    lon_name: str
        name of the longitude coordinate
    roll_by: int
        number of grid points to roll the data by - must be more than the gap
    **kwargs:
        passed to xr.interp
    """
    interpolated = (
        ds
        .interp(**kwargs)
        .roll(**{lon_name: roll_by}, roll_coords=False)
        .interpolate_na(lon_name, limit=int(roll_by / 2))
        .roll(**{lon_name: -roll_by}, roll_coords=False)
    )

    return interpolated
