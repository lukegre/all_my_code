"""
Used to process netCDF files to a standard format.
All functions take in an xarray.Dataset and return an xarray.Dataset
All functions in this preprocessing module should add metadata to the
xarray object under `history`
"""


def rename_to_latlon(xds):
    """
    Tries to rename latitude to lat and longitude to lon
    from a list of standard names:
        time: mtime
        lat: latitude, lats, yt_ocean
        lon: longitude, lons, longs, long, xt_ocean
    """

    time = ['mtime']
    lat = ['latitude', 'lats', 'yt_ocean']
    lon = ['longitude', 'lons', 'xt_ocean', 'longs', 'long']

    rename_dict = {}
    for key in xds.coords.keys():
        if key in time:
            rename_dict[key] = 'time'
        if key in lat:
            rename_dict[key] = 'lat'
        if key in lon:
            rename_dict[key] = 'lon'

    xds = xds.rename(rename_dict)
    if rename_dict:
        xds = _netcdf_add_brew_hist(xds, 'renamed time lats and lons')

    return xds


def center_coords_at_0(xds):
    """
    Change longitude from 0:360 to -180:180
    """
    import numpy as np

    def strictly_increasing(L):
        return all([x < y for x, y in zip(L, L[1:])])

    x = xds['lon'].values
    y = xds['lat'].values

    x[x >= 180] -= 360
    if not strictly_increasing(x):
        sort_idx = np.argsort(x)
        xds = xds.isel(**{'lon': sort_idx})
        xds['lon'].values = x[sort_idx]
        xds = _netcdf_add_brew_hist(xds, 'center coords -> 0:360 to -180:180')

    if not strictly_increasing(y):
        xds = xds.isel(**{'lat': slice(None, None, -1)})
        xds = _netcdf_add_brew_hist(xds, 'flipped lats to -90:90')

    return xds


def center_time_monthly_15th(xds):
    """
    If monthly data, centeres on the 15th of the month
    """
    from pandas import Timestamp

    assert xds.time.size == 1, 'Only accepts one month DataArrays'

    attrs = xds.attrs
    time = Timestamp(xds.time.values[0])

    xds.time.values[:] = Timestamp(year=time.year, month=time.month, day=15)

    xds.attrs = attrs
    xds = _netcdf_add_brew_hist(xds, 'centered monthly data to 15th')
    return xds


def shallowest(xda):
    """
    BROKEN

    Gets the surface data
    """
    for dim in xda.dims:
        depth_dim = None
        if hasattr(xda[dim], 'units'):
            units = xda[dim].units
            if 'meters' in units:
                depth_dim = dim
                break
    if depth_dim is None:
        return xda

    xda = xda.sel(**{depth_dim: 0}, method='nearest').drop(depth_dim)
    xda = _netcdf_add_brew_hist(xda, 'Shallowest depth selected')
    return xda


def interpolate_025(xds, method='linear'):
    """
    Interpolates global data linearly, but I recommend rather using CDO's
    bilinear interpolation
    """
    from numpy import arange

    attrs = xds.attrs
    xds = (
        xds.interp(
            lat=arange(-89.875, 90, 0.25),
            lon=arange(-179.875, 180, 0.25),
            method=method,
        )
        # filling gaps due to interpolation along 180deg
        .roll(lon=720, roll_coords=False)
        .interpolate_na(dim='lon', limit=10)
        .roll(lon=-720, roll_coords=False)
    )

    xds.attrs = attrs
    xds = _netcdf_add_brew_hist(xds, 'interpolated to 0.25deg')

    return xds


def interpolate_1deg(xds, method='linear'):
    from numpy import arange

    attrs = xds.attrs
    xds = (
        xds.interp(
            lat=arange(-89.5, 90), lon=arange(-179.5, 180), method=method
        )
        # filling gaps due to interpolation along 180deg
        .roll(lon=180, roll_coords=False)
        .interpolate_na(dim='lon', limit=3)
        .roll(lon=-180, roll_coords=False)
    )

    xds.attrs = attrs
    xds = _netcdf_add_brew_hist(xds, 'interpolated to 1deg')

    return xds


def resample_time_1D(xds):
    attrs = xds.attrs

    xds = xds.resample(time='1D', keep_attrs=True).mean(
        'time', keep_attrs=True
    )

    xds.attrs.update(attrs)
    xds = _netcdf_add_brew_hist(xds, 'resampled to time to 1D')

    return xds


def resample_time_1M(xds):
    import pandas as pd

    attrs = xds.attrs

    xds = xds.resample(time='1MS', keep_attrs=True).mean(
        'time', keep_attrs=True
    )
    xds.time.values += pd.Timedelta('14D')

    xds.attrs.update(attrs)
    xds = _netcdf_add_brew_hist(xds, 'resampled to time to 1M')

    return xds


def fill_time_monthly_to_daily(xds):
    import pandas as pd

    time = xds.time.to_index()
    year_0 = time.year.unique().values[0]
    mon_0 = time.month.unique().values[0]

    year_1 = year_0 if mon_0 < 12 else year_0 + 1
    mon_1 = 1 if mon_0 == 12 else mon_0 + 1

    t0 = pd.Timestamp(f'{year_0}-{mon_0}-01')
    t1 = pd.Timestamp(f'{year_1}-{mon_1}-01')

    date_range = pd.date_range(start=t0, end=t1, freq='1D', closed='left')
    xds = xds.reindex(time=date_range, method='nearest')

    xds = _netcdf_add_brew_hist(xds, 'time filled from monthly to daily')

    return xds


def unzip(zip_path, dest_dir=None, verbose=1):
    """returns a list of unzipped file names"""
    import os
    from zipfile import ZipFile

    def get_destination_directory(zipped):
        file_name = zipped.filename
        file_list = zipped.namelist()
        if len(file_list) == 1:
            destdir = os.path.split(file_name)[0]
        else:
            destdir = os.path.splitext(file_name)[0]

        return destdir

    def get_list_of_zipped_files(zipped, dest_dir):
        flist_zip = set(zipped.namelist())
        flist_dir = set(os.listdir(dest_dir))

        for file in flist_dir:
            if not is_local_file_valid(file):
                flist_dir -= set(file)

        files_to_extract = list(flist_zip - flist_dir)

        if not files_to_extract:
            if verbose:
                print(f'All files extracted: {zipped.filename}')
        return files_to_extract

    if not os.path.isfile(zip_path):
        raise OSError(f'The zip file does not exist: {zip_path}')

    zipped = ZipFile(zip_path, 'r')
    if dest_dir is None:
        dest_dir = get_destination_directory(zipped)
        os.makedirs(dest_dir, exist_ok=True)

    files_to_extract = get_list_of_zipped_files(zipped, dest_dir)
    for file in files_to_extract:
        if verbose:
            print(f' Extracting: {file}')
        zipped.extractall(path=dest_dir, members=[file])

    return [os.path.join(dest_dir, f) for f in zipped.namelist()]


def gunzip(zip_path, dest_path=None):
    import shutil
    import gzip

    if dest_path is None:
        dest_path = zip_path.replace('.gz', '')

    with gzip.open(zip_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            return f_out


def untar(tar_path, dest_dir=None, verbose=1):
    """returns a list of untarred file names"""
    import os
    import pathlib
    import tarfile

    if not os.path.isfile(tar_path):
        raise OSError(f'The tar file does not exist: {tar_path}')

    if tar_path.endswith('gz'):
        mode = 'r:gz'
    else:
        mode = 'r:'
    tarred = tarfile.open(tar_path, mode)

    if dest_dir is None:
        dest_dir = pathlib.Path(tar_path).parent
    else:
        os.makedirs(dest_dir, exist_ok=True)

    tarred.extractall(path=dest_dir)

    return [os.path.join(dest_dir, f) for f in tarred.getnames()]


def _netcdf_add_brew_hist(xds, msg, key='history'):
    from pandas import Timestamp

    now = Timestamp.today().strftime('%Y-%m-%dT%H:%M')
    prefix = f'\n[DataBrewery@{now}] '
    msg = prefix + msg
    if key not in xds.attrs:
        xds.attrs[key] = msg
    elif xds.attrs[key] == '':
        xds.attrs[key] = msg
    else:
        xds.attrs[key] += '; ' + msg

    return xds


def apply_process_pipeline(xds, pipe):
    """
    Applies a list of functions to an xarray.Dataset object.
    Functions must accept a Dataset and return a Dataset
    """
    attrs = xds.attrs
    for func in pipe:
        xds = func(xds)

    xds.attrs = attrs
    return xds


def is_local_file_valid(local_path):
    from os.path import isfile

    if not isfile(local_path):
        return False

    # has an opener been passed, if not assumes file is valid
    if local_path.endswith('.nc'):
        from netCDF4 import Dataset as opener

        error = OSError
    elif local_path.endswith('.zip'):
        from zipfile import ZipFile as opener, BadZipFile as error
    else:
        error = BaseException

        def opener(p):
            return None  # dummy opener

    # tries to open the path, if it fails, not valid, if it passes, valid
    try:
        with opener(local_path):
            return True
    except error:
        return False
