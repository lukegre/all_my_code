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


def pca_decomp(
    xda, n_components=10, return_plots=False, return_pca=False, **pca_kwargs,
):
    """
    Apply a principle component decomposition to a dataset with
    time, lat, lon axes.
    """
    from sklearn.decomposition import PCA

    def unnan(arr):
        t, y, x = arr.shape
        flat = arr.reshape(t, -1)
        mask = ~np.isnan(flat).any(0)
        return flat[:, mask], mask

    def renan(arr, mask, shape=None):
        out = np.ndarray([min(arr.shape), mask.size]) * np.NaN
        if np.argmin(arr.shape) == 1:
            arr = arr.T
        out[:, mask] = arr
        out = out
        if shape:
            out = out.reshape(*shape)
        return out

    t, y, x = xda.dims

    assert t.lower() in [
        "time",
        "date",
        "tmnth",
        "days",
    ], "DataArray needs to have time as first dimension"
    assert (
        y.lower() in "ylatitude"
    ), "DataArray needs to have latitude as second dimension"
    assert (
        x.lower() in "xlongitude"
    ), "DataArray needs to have longitude as third dimension"

    coords = {d: xda[d].values for d in xda.dims}
    coords.update({"n_components": np.arange(n_components)})

    pca = PCA(n_components=n_components, **pca_kwargs)

    v, m = unnan(xda.values)

    trans = pca.fit_transform(v.T)
    trans_3D = renan(trans, m, shape=[n_components, coords[y].size, coords[x].size])

    xds = xr.Dataset(attrs={"name": xda.name})
    dims = ["n_components", "lat", "lon"]
    props = dict(coords={k: coords[k] for k in dims}, dims=dims)
    xds["transformed"] = xr.DataArray(trans_3D, **props)

    dims = ["n_components", "time"]
    props = dict(coords={k: coords[k] for k in dims}, dims=dims)
    xds["principle_components"] = xr.DataArray(pca.components_, **props)

    dims = ["time"]
    props = dict(coords={k: coords[k] for k in dims}, dims=dims)
    xds["mean_"] = xr.DataArray(pca.mean_, **props)

    dims = ["n_components"]
    props = dict(coords={k: coords[k] for k in dims}, dims=dims)
    xds["variance_explained"] = xr.DataArray(pca.explained_variance_ratio_, **props)

    if return_plots and return_pca:
        fig = _pca_plot(xds)
        return xds, pca, fig
    elif return_plots:
        fig = _pca_plot(xds)
        return xds, fig
    elif return_pca:
        return xds, pca
    else:
        return xds


def _pca_plot(xds_pca):
    from matplotlib import pyplot as plt

    n = xds_pca.n_components.size
    fig = plt.figure(figsize=[15, n * 3.2], dpi=120)
    shape = n, 5
    ax = []

    for i in range(shape[0]):
        ax += (
            [
                plt.subplot2grid(shape, [i, 0], colspan=3, fig=fig),
                plt.subplot2grid(
                    shape, [i, 3], colspan=2, fig=fig, facecolor="#AAAAAA"
                ),
            ],
        )

    t = xds_pca.principle_components.dims[-1]
    y, x = xds_pca.transformed.dims[1:]
    for i in xds_pca.n_components.values:
        pt = xds_pca[t].values
        px = xds_pca[x].values
        py = xds_pca[y].values
        pz = xds_pca.transformed[i].to_masked_array()

        var = xds_pca.variance_explained[i].values * 100
        lim = np.nanpercentile(abs(pz.filled(np.nan)), 99)

        a0 = ax[i][0]
        a1 = ax[i][1]

        a0.plot(pt, xds_pca.principle_components[i].values)
        a0.axhline(0, color="k")
        a0.set_ylabel("Component {}\n({:.2f}%)".format(i + 1, var), fontsize=12)

        img = a1.pcolormesh(
            px, py, pz, vmin=-lim, rasterized=True, vmax=lim, cmap=plt.cm.RdBu_r,
        )
        plt.colorbar(img, ax=a1)
        img.colorbar.set_label("Transformed units")

        if i != (shape[0] - 1):
            a0.set_xticklabels([])
            a1.set_xticklabels([])
        else:
            pass

    title = (
        "Principle Component Analysis (PCA) "
        "for {} showing the first {} components"
    )
    fig.suptitle(
        title.format(xds_pca.name, n),
        y=1.01,
        x=0.5,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()

    return fig