import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy import convolution as conv

warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", ".*invalid value encountered in less.*")
warnings.filterwarnings("ignore", ".*convolution.*")


@xr.register_dataarray_accessor("stats")
class Statistics(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def pca_decomp(
        self, n_components=10, return_plots=False, return_pca=False, **pca_kwargs,
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

        xda = self._obj
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
            fig = self._pca_plot(xds)
            return xds, pca, fig
        elif return_plots:
            fig = self._pca_plot(xds)
            return xds, fig
        elif return_pca:
            return xds, pca
        else:
            return xds

    @staticmethod
    def _pca_plot(xds_pca):
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

    def trend(
        self, dim=None, return_stats=True, return_trend=True, return_input=False,
    ):
        """
        Calculates the trend of the data along the first or given dimension
        of the input data array. Uses y = mx + c

        Parameters
        ----------
        dim: str
            calculate the trend along the given dimension. If left as None,
            will assume that the first dimension is the dimension along which
            you want to calculate the trend
        return_stats: bool
            when set to True, will return the intercept, slope and pvalues
        return_trend: bool
            when set to True, will return the trend data with the same shape as
            the input

        Returns
        -------
        trend_data : xr.Dataset
            A dataset containing the slope, intercept and p-values
            If trend is requested, then the calculated slope is included
        """
        from .date_utils import convert_time_to_most_suitable_unit

        xda = self._obj

        assert isinstance(dim, str) | (dim is None), "'dim' must be str or None"
        assert isinstance(return_trend, bool), "return_trend must be boolean"
        assert isinstance(return_stats, bool), "return_stats must be boolean"
        assert (
            return_stats | return_trend
        ), "no point in running the function when you don't want stats or trends"

        if dim is None:
            dim = xda.dims[0]
        elif dim not in xda.dims:
            raise KeyError(f"'{dim}' is not a dim in the list of dims {xda.dims}")

        mask = xda.notnull()
        # xda = xda.where(mask, drop=True)
        # getting shapes
        # creating x and y variables for linear regression
        x_optimal_unit = convert_time_to_most_suitable_unit(xda[dim])
        x = xr.DataArray(x_optimal_unit.astype(float), dims=(dim,))
        y = xda.where(mask)

        # ############################ #
        # LINEAR REGRESSION DONE BELOW #
        xm = x.mean()  # mean
        ym = xda.mean(dim)  # mean
        ya = y - ym  # anomaly
        xa = x - xm  # anomaly

        # variance and covariances
        xss = (xa ** 2).sum(dim)  # variance of x (with df as n-1)
        yss = (ya ** 2).sum(dim)  # variance of y (with df as n-1)
        xys = (xa * ya).sum(dim)  # covariance    (with df as n-1)
        # slope and intercept
        slope = xys / xss
        intercept = ym - (slope * xm)

        # preparing outputs
        name = xda.name if not hasattr(xda, "name") else "array"
        if name is None:
            name = "array"
        out = xda.to_dataset(name=name)
        # dummy = xda.isel(**{dim: slice(0, 2)}).mean(dim)
        units = xda.attrs["units"] if "units" in xda.attrs else ""

        dim_unit = str(x_optimal_unit.dtype)
        time_unit_abbrev = dict(Y="year", M="month", D="day", m="minute", s="second")
        if "datetime64" in dim_unit:
            dim_unit = re.sub("datetime64|\[|\]", "", dim_unit)
            dim_unit = time_unit_abbrev.get(dim_unit, dim_unit)
        else:
            dim_unit = f"{dim}_step"

        if return_stats:
            from scipy import stats

            # statistics about fit
            df = x.count(dim) - 2
            r = xys / (xss * yss) ** 0.5
            t = r * (df / ((1 - r) * (1 + r))) ** 0.5
            p = stats.distributions.t.sf(abs(t), df)

            # first create variable for slope and adjust meta
            out["slope"] = slope
            out["slope"].name += "_slope"
            out["slope"].attrs["units"] = f"{units} / {dim_unit}"
            # out["slope"].values = slope.reshape(shape)

            # first create variable for slope and adjust meta
            out["intercept"] = intercept  # dummy.copy()
            out["intercept"].name += "_intercept"
            out["intercept"].attrs["units"] = units
            # out["intercept"].values = intercept.reshape(shape)

            # do the same for the p value
            out["pval"] = xr.DataArray(
                p, coords=out.slope.coords, dims=out.slope.dims
            )  # dummy.copy()
            out["pval"].name += "_Pvalue"
            # out["pval"].values = p.reshape(shape)
            out["pval"].attrs["info"] = (
                "If p < 0.05 then the results " "from 'slope' are significant."
            )
            # out["pval"] = out.pval.where(out.slope.notnull())

        if return_trend:
            # from numpy import dot
            yhat = (slope * x + intercept).transpose(*xda.dims)
            out["trend"] = yhat  # xda.copy()
            out["trend"].name += "_trend"
            out["trend"].attrs["units"] = f"{units}"
            # out["trend"].values = yhat.reshape(xda.shape)

        out[dim] = x_optimal_unit
        out[dim].attrs["units"] = dim_unit

        if not return_input:
            out = out.drop(name)

        return out.reindex(**{dim: mask[dim]})

    def detrend(self, dim=None):
        """
        Removes the trend of the data along the first or given dimension
        of the input data array. Uses y = mx + c

        Parameters
        ----------
        dim: str
            calculate the trend along the given dimension. If left as None,
            will assume that the first dimension is the dimension along which
            you want to calculate the trend

        Returns
        -------
        trend_data : xr.DataArray
            A data array that is the same as the input, but without the linear
            trend along the given dimension
        """

        xda = self._obj
        trend = self.trend(dim=dim, return_trend=True, return_stats=False).trend

        name = xda.name if hasattr(xda, "name") else "array"
        name = "array" if name is None else name
        detrended = xda - trend
        detrended.name = name + "detrended"
        detrended.attrs[
            "description"
        ] = f"linearly detrended data along the {dim} dimension."

        return detrended

    def corr_vars(self, xarr2):
        from pandas import DataFrame

        xarr1 = self._obj.copy()
        assert (
            xarr1.shape == xarr2.shape
        ), "The input DataArray must be the same size as {}".format(self.name)

        xarr3 = xarr1[:1].mean("time").copy()

        t, y, x = xarr1.shape

        df1 = DataFrame(xarr1.values.reshape(t, y * x))
        df2 = DataFrame(xarr2.values.reshape(t, y * x))

        dfcor = df1.corrwith(df2).values.reshape(y, x)
        xarr3.values = dfcor

        xarr3.attrs["long_name"] = "Correlation of %s and %s" % (
            xarr1.name,
            xarr2.name,
        )
        xarr3.name = "corr_%s_vs_%s" % (xarr1.name, xarr2.name)

        xarr3.encoding.update({"zlib": True, "shuffle": True, "complevel": 4})

        return xarr3


@xr.register_dataarray_accessor("average")
class Average:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, dim=None, axis=None, weights=None):
        xda = self._obj
        return self._average(xda, dim=dim, axis=axis, weights=weights)

    @staticmethod
    def _average(xda, dim=None, axis=None, weights=None):
        """
        dim = dimension to average over
        weights = xdawith the same dimension and weights
        """
        if weights is None:
            return xda.mean(dim=dim, axis=axis)
        else:
            print("doing weighted average")
            a = (xda * weights).sum(dim=dim, axis=axis)
            b = (xda.notnull() * weights).sum(dim=dim, axis=axis)
            return a / b


@xr.register_dataarray_accessor("climatology")
@xr.register_dataset_accessor("climatology")
class Climatology:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def climatology(self, full=False, period="month", dim="time"):
        from warnings import filterwarnings

        filterwarnings("ignore", ".*Slicing with an.*")

        xro = self._obj
        if isinstance(xro, xr.DataArray):
            return self._climatology(xro, full=full, period=period, dim=dim)
        else:
            return xro.apply(self._climatology, full=full, period=period, dim=dim)

    __call__ = climatology

    def anomaly(self, period="month", dim="time"):
        from warnings import filterwarnings

        filterwarnings("ignore", ".*Slicing with an.*")

        xro = self._obj
        if isinstance(xro, xr.DataArray):
            return self.climatology(xro, period=period, dim=dim)
        else:
            return xro.apply(self.climatology, period=period, dim=dim)

    @staticmethod
    def _climatology(xda, full=False, period="month", dim="time", reduce_func="mean"):
        group = xda.groupby(f"{dim}.{period}")
        clim = getattr(group, reduce_func)(dim)
        if full:
            return (
                clim.sel(month=getattr(xda[dim].to_index(), period))
                .rename({period: dim})
                .assign_coords(**{dim: xda[dim]})
            )
        else:
            return clim

    @staticmethod
    def _anomaly(xda, period="month", dim="time"):
        group = xda.groupby(f"{dim}.{period}")
        return group - group.mean(dim)


@xr.register_dataarray_accessor("convolve")
class Convolve(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(
        self, kernel=conv.Gaussian2DKernel(x_stddev=2), fill_nans=False, verbose=True,
    ):
        return self.spatial(kernel, fill_nans, verbose)

    @staticmethod
    def _convlve_timestep(xda, kernel, preserve_nan):
        convolved = xda.copy()
        convolved.values = conv.convolve(
            xda.values, kernel, preserve_nan=preserve_nan, boundary="wrap"
        )
        return convolved

    def spatial(self, kernel=None, fill_nans=False, verbose=True):
        xda = self._obj
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
            convolved = self._convlve_timestep(xda, kernel, preserve_nan)
        elif ndims == 3:
            convolved = []
            for t in range(xda.shape[0]):
                if verbose:
                    print(".", end="")
                convolved += (self._convlve_timestep(xda[t], kernel, preserve_nan),)
            convolved = xr.concat(convolved, dim=xda.dims[0])

        kern_size = kernel.shape
        convolved.attrs["description"] = (
            "same as `{}` but with {}x{}deg (lon x lat) smoothing using "
            "astropy.convolution.convolve"
        ).format(xda.name, kern_size[0], kern_size[1])
        return convolved


@xr.register_dataarray_accessor("fill_empty_diff")
class FillEmpty(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, filler):
        xda = self._obj
        return self._fill_empty_diff(xda, filler)

    @staticmethod
    def _fill_empty_diff(xda, filler):
        assert xda.shape == filler.shape, "both arrays must have the same shape"

        mask = (xda.isnull() & filler.notnull()).values
        arr = xda.values.copy()
        arr[mask] = filler.values[mask]

        xda.values = arr
        return xda
