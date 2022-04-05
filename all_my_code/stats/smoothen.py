from xarray import register_dataarray_accessor as _register_dataarray_accessor
from functools import wraps


def _interp_xarray_with_scipy_interp(da, interp_func, dim='time', lengthening_factor=10, **kwargs):
    """
    a helper function to interpolate data arrays
    """
    from numpy import linspace
    from xarray import DataArray

    assert isinstance(da, DataArray), 'input must be a data array'
    assert len(da.dims) == 1, 'only 1-dimensional arrays supported at the moment'
    assert lengthening_factor > 0, 'the lengthening_factor must be greater than 0'
    
    y = da.values
    
    # if x is datetime64, this stores the type to reconvert later
    x_dtype = da[dim].values.dtype
    # now convert x to float
    x = da[dim].values.astype(float)
    # create the output x-dim
    x_interp = linspace(x.min(), x.max(), x.size * lengthening_factor)
    
    # fit the function so that we can predict
    fitted = interp_func(x, y, **kwargs)
 
    # the smoothed output is put back into a data array
    predicted = DataArray(
        fitted(x_interp), 
        coords={dim: x_interp.astype(x_dtype)},
        dims=da.dims)
    
    return predicted


def spline(da, degree=2, dim='time', lengthening_factor=10, **spline_kwargs):
    """
    A wrapper around scipy.interpolate.make_interp_spline to 
    quickly create a smooth spline from a single dimension data. 
    
    Compared with rolling mean approaches, splines will interpolate
    beyond the data, thus increasing the min and max. 
    
    Parameters
    ----------
    da : xr.DataArray
        a single dimension xarray.DataArray. Note that this does 
        not have to be given if using the method approach. 
    degree : int [2]
        the degree of the spline that you'd like to fit
    dim : str [time]
        the deminsion along which the interpolation will take place
    lengthening_factor : int [10]
        the output array will be longer by this factor. a number that
        is too low will not result in a smooth spline
    spline_kwargs : dict
        keyword-value pairs for the make_interp_spline function 
        
    Returns
    -------
    xr.DataArray : 
        a smooth data array that is longer than the input by a 
        factor of the lengthening factor
        
    """
    from scipy.interpolate import make_interp_spline
    
    spline_kwargs.update(k=degree)
    smooth = _interp_xarray_with_scipy_interp(
        da, 
        make_interp_spline, 
        dim=dim, 
        lengthening_factor=lengthening_factor,
        **spline_kwargs
    )
    
    smooth = smooth.assign_attrs(da.attrs)
    
    return smooth


def rolling_ewm(da, radius=0.5, fill_tail=True, lengthening_factor=10, dim='time', **interp1d_kwargs):
    """
    Smoothens off the corners of a line. 
    
    Compared with spline approaches, this approach will round the 
    corners of the line thus reducing the min and max. 
    beyond the data. 
    
    Parameters
    ----------
    da : xr.DataArray
        a single dimension xarray.DataArray. Note that this does 
        not have to be given if using the method approach. 
    radius : float [0.5]
        the radius of the exponentially weighted mean window relative
        to the length of the original input. 
    fill_tail : bool [True]
        will fill the last few points of the input with the original data
        if set to true, otherwise, will return the "valid" output
    dim : str [time]
        the deminsion along which the interpolation will take place
    lengthening_factor : int [10]
        the output array will be longer by this factor. a number that
        is too low will not result in a smooth spline
        
    Returns
    -------
    xr.DataArray : 
        a smooth data array that is longer than the input by a 
        factor of the lengthening factor
    """
    from scipy.interpolate import interp1d
    
    window = int(radius * lengthening_factor)
    if not window % 2:
        window += 1

    linear_interp = _interp_xarray_with_scipy_interp(
        da, 
        interp1d, 
        dim=dim, 
        lengthening_factor=lengthening_factor,
        **interp1d_kwargs
    )
    
    # if the window is set to 0, simply return the lengthened array
    if window == 0:
        return linear_interp
    
    # the smoothing part
    smooth = linear_interp.rolling_exp(**{dim: window}).mean()
    
    # the rolling_exp function does not center the dates, so we do that here
    shift = -int(window / 2)
    shifted = smooth.roll(**{dim: shift})[:shift].reindex_like(linear_interp)
    
    # filling the missing ends with the linear interpolated results
    if fill_tail:
        shifted = shifted.fillna(linear_interp)
    
    return shifted


def loess(da, data_frac=2/3, dim='time', lengthening_factor=1, **loess_kwargs):
    """
    Performs a loess regression to find the data trend
    
    An xarray wrapper around statsmodels.api.nonparametric.lowess
    
    Parameters
    ----------
    da : xr.DataArray
        a single dimension xarray.DataArray. Note that this does 
        not have to be given if using the method approach. 
    data_frac : float [0.66666]
        The fraction of the total dataset used to calculate the trend 
    dim : str [time]
        the deminsion along which the interpolation will take place
    lengthening_factor : int [1]
        by default, the dataset is not lengthened, but can be lengthened
        by this factor if required
    loess_kwargs : dict
        input options for the loess function. Note that `data_frac` 
        will always overwrite `frac`. 
        
    Returns
    -------
    xr.DataArray : 
        a smooth data array that is longer than the input by a 
        factor of the lengthening factor
    
    """
    from xarray import DataArray
    from scipy.interpolate import interp1d
    from statsmodels import api as sm
    
    if lengthening_factor > 1:
        da = _interp_xarray_with_scipy_interp(
            da, 
            interp1d, 
            dim=dim, 
            lengthening_factor=lengthening_factor)
    
    y = da.values

    x_dtype = da[dim].values.dtype
    x = da[dim].values.astype(float)

    loess_kwargs.update(is_sorted=False, return_sorted=True, frac=data_frac)
    out = sm.nonparametric.lowess(y, x, **loess_kwargs).T
    
    smoothed = DataArray(
        out[1],
        coords={dim: out[0].astype(x_dtype)},
        dims=da.dims)
    
    return smoothed


def convolve(da, kernel=None, fill_nans=False, verbose=True):
    from astropy import convolution as conv
    
    def _convlve2D(xda, kernel, preserve_nan):
        convolved = xda.copy()
        convolved.values = conv.convolve(
            xda.values, kernel, preserve_nan=preserve_nan, boundary="wrap"
        )
        return convolved
    ndims = len(da.dims)
    preserve_nan = not fill_nans

    if kernel is None:
        kernel = conv.Gaussian2DKernel(x_stddev=2)
    elif isinstance(kernel, list):
        if len(kernel) == 2:
            kernel_size = kernel
            for i, ks in enumerate(kernel_size):
                kernel_size[i] += 0 if (ks % 2) else 1
            kernel = conv.kernels.Box2DKernel(max(kernel_size))
            kernel._array = kernel._array[: kernel_size[0], : kernel_size[1]]
        else:
            raise UserWarning(
                "If you pass a list to `kernel`, must have a length of 2"
            )
    elif kernel.__class__.__base__ == conv.core.Kernel2D:
        kernel = kernel
    else:
        raise UserWarning(
            "kernel needs to be list or astropy.kernels.Kernel2D base type"
        )

    if ndims == 2:
        convolved = _convlve2D(da, kernel, preserve_nan)
    elif ndims == 3:
        convolved = []
        for t in range(da.shape[0]):
            if verbose:
                print(".", end="")
            convolved += (_convlve2D(da[t], kernel, preserve_nan),)
        convolved = xr.concat(convolved, dim=da.dims[0])

    kern_size = kernel.shape
    convolved.attrs["description"] = (
        "same as `{}` but with {}x{}deg (lon x lat) smoothing using "
        "astropy.convolution.convolve"
    ).format(da.name, kern_size[0], kern_size[1])
    return convolved


@_register_dataarray_accessor('smooth')
class Smooth(object):
    def __init__(self, da):
        self._obj = da
    
    @wraps(spline)
    def spline(self, **kwargs):
        return spline(self._obj, **kwargs)

    @wraps(rolling_ewm)
    def ewm(self, **kwargs):
        return rolling_ewm(self._obj, **kwargs)
    
    @wraps(loess)
    def loess(self, **kwargs):
        return loess(self._obj, **kwargs)
    
    @wraps(convolve)
    def convolve(self, **kwargs):
        return convolve(self._obj, **kwargs)
