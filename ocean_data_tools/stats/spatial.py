import xarray as xr
import numpy as np


def regional_aggregation(xda, region_mask, weights=None, func='mean'):
    
    regional = []
    for r, da in xda.groupby(region_mask):
        da = da.unstack()
        if weights is not None:
            da = da.weighted(weights)
        da = getattr(da, func)(['lat', 'lon'])
        da = da.assign_coords(region=r)
        regional += da,
        
    regional = xr.concat(regional, 'region')
    return regional
    

def convolve(da, kernel=None, fill_nans=False, verbose=True):
    from astropy import convolution as conv
    
    def _convlve2D(xda, kernel, preserve_nan):
        convolved = xda.copy()
        convolved.values = conv.convolve(
            xda.values, kernel, preserve_nan=preserve_nan, boundary="wrap"
        )
        return convolved
    ndims = len(xda.dims)
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
        convolved = _convlve2D(xda, kernel, preserve_nan)
    elif ndims == 3:
        convolved = []
        for t in range(xda.shape[0]):
            if verbose:
                print(".", end="")
            convolved += (_convlve2D(xda[t], kernel, preserve_nan),)
        convolved = xr.concat(convolved, dim=xda.dims[0])

    kern_size = kernel.shape
    convolved.attrs["description"] = (
        "same as `{}` but with {}x{}deg (lon x lat) smoothing using "
        "astropy.convolution.convolve"
    ).format(xda.name, kern_size[0], kern_size[1])
    return convolved

