from scipy.stats import distributions as dist
import numpy as np
import matplotlib.pyplot as plt


def get_distribution_fit(y, bins=80, dist_func=dist.norm, **fit_kwargs):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    args = dist_func.fit(y, **fit_kwargs)
    fitted_dist = dist_func(*args)

    if isinstance(bins, int):
        xbins = np.linspace(y.min(), y.max(), bins + 1)
    else:
        xbins = bins

    x = np.convolve(xbins, [0.5, 0.5], mode="valid")
    yhst = np.histogram(y, bins=xbins, density=True)[0]
    yhat = fitted_dist.pdf(x)

    resid = yhst - yhat
    rmse = (resid**2).mean() ** 0.5

    return dict(dist=fitted_dist, args=args, x=x, yhat=yhat, yhst=yhst, rmse=rmse)


def plot_distribution(
    y=None, bins=30, dist_func=dist.norm, ax=None, annot=True, **kwargs
):

    args = dist_func.fit(y)
    dist = dist_func(*args)

    if ax is None:
        fig, ax = plt.subplots()

    ybin, xbin, _ = ax.hist(y, bins=bins, density=True, color="k", lw=0)

    x = np.linspace(xbin.min(), xbin.max(), 100)
    yhat = dist.pdf(np.convolve(xbin, [0.5] * 2, mode="valid"))
    yhat_plot = dist.pdf(x)
    ax.plot(x, yhat_plot)

    description = {
        "name": dist_func.__class__.__name__.replace("_gen", ""),
        "args": args,
        "rmse": ((ybin - yhat) ** 2).mean() ** 0.5,
        "mean": dist.stats(moments="m"),
        "var": dist.stats(moments="v"),
        "mode": x[yhat_plot.argmax()],
    }

    txt = ""
    for key, val in description.items():
        if isinstance(val, (float, np.ndarray)):
            txt += f"{key} = {val:.2f}\n"
        elif isinstance(val, (list, tuple)):
            val = [np.around(v, 2) for v in val]
            txt += f"{key} = {val}\n"
        else:
            txt += f"{key} = {val}\n"
    txt = txt.replace("[", "").replace("]", "")

    if annot:
        ax.text(0.95, 0.95, txt, ha="right", va="top", transform=ax.transAxes)

    if hasattr(y, "name"):
        ax.set_title(y.name)

    ax.description = description

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
        d.chi2,
        d.expon,
        d.f,
        d.gamma,
        d.genextreme,
        d.gumbel_r,
        d.laplace,
        d.logistic,
        d.lognorm,
        d.norm,
        d.pareto,
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
            name = d.__class__.__name__.replace("_gen", "")
            props = fit_kwargs
            props.update(dict(dist_func=d))
            results[name] = get_distribution_fit(data, **props)
            print("", end=".")
        except KeyboardInterrupt:
            raise KeyboardInterrupt

    print()

    return results


def get_best_distributions_for_df(df, **kwargs):
    results = {}
    for col in df:
        print(col, end="")
        results[col] = find_best_distribution_fit(df[col], **kwargs)
        print()
    return results
