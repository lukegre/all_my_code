from scipy.stats import distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_distribution_fit(y, n_bins=80, dist_func=dist.norm, **fit_kwargs):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    args = dist_func.fit(y, **fit_kwargs)
    fitted_dist = dist_func(*args)

    xbins = np.linspace(y.min(), y.max(), n_bins + 1)
    x = np.convolve(xbins, [0.5, 0.5], mode="valid")
    yhst = np.histogram(y, bins=xbins, density=True)[0]
    yhat = fitted_dist.pdf(x)

    resid = yhst - yhat
    rmse = (resid**2).mean() ** 0.5

    return dict(dist=fitted_dist, args=args, x=x, yhat=yhat, yhst=yhst, rmse=rmse)


def plot_distribution(y=None, n_bins=30, dist_func=dist.norm, ax=None, **kwargs):

    args = dist_func.fit(y)
    dist = dist_func(*args)

    x = np.linspace(y.min(), y.max(), 100)
    yhat_plot = dist.pdf(x)

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(y, bins=n_bins, density=True, color="k")
    ax.plot(x, yhat_plot)

    txt = "\n".join(np.array(args).round(4).astype(str))
    ax.text(0.95, 0.95, txt, ha="right", va="top", transform=ax.transAxes)

    return ax


def list_all_scipy_distributions():
    distributions = [
        f
        for f in dir(dist._continuous_distns)
        if (not f.startswith("_") and (f.lower() == f))
    ]
    distributions.remove("levy_stable")
    distributions.remove("studentized_range")

    dists = [getattr(dist, d) for d in distributions]

    return dists


def common_distributions():
    from scipy.stats import distributions as d

    return [
        d.alpha,
        d.beta,
        d.binom,
        d.chi2,
        d.expon,
        d.f,
        d.gamma,
        d.geom,
        d.hypergeom,
        d.laplace,
        d.logistic,
        d.lognorm,
        d.nbinom,
        d.norm,
        d.pareto,
        d.poisson,
        d.powerlaw,
        d.rayleigh,
        d.skewnorm,
        d.t,
        d.uniform,
        d.weibull_max,
        d.weibull_min,
    ]


def find_best_distribution_fit(
    data, distributions=common_distributions(), **fit_kwargs
):
    if distributions is None:
        distributions = list_all_scipy_distributions()

    results = {}
    for d in distributions:
        try:
            name = d.__class__.__name__.relace("_gen", "")
            props = fit_kwargs
            props.update(dict(dist_func=d, plot=0))
            results[name] = get_distribution_fit(data, **props)
            print("", end=".")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
            pass

    s = pd.DataFrame(results)
    return s


def get_best_distributions_for_df(df, **kwargs):
    results = {}
    for col in df:
        print(col, end="")
        results[col] = find_best_distribution_fit(df[col], **kwargs)
        print()
    return results
