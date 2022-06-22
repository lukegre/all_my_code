from scipy.stats import distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def fit_distribution(y, n_bins=80, dist_func=dist.norm, plot=False, metric="rmse"):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    args = dist_func.fit(y)
    dist = dist_func(*args)

    if not plot:
        xbins = np.linspace(y.min(), y.max(), n_bins + 1)
        x = np.convolve(xbins, [0.5, 0.5], mode="valid")
        yhst = np.histogram(y, bins=xbins, density=True)[0]
        yhat = dist.pdf(x)

        resid = yhst - yhat
        if metric == "rmse":
            result = (resid**2).mean() ** 0.5
        elif metric == "mae":
            result = abs(resid).mean()
        else:
            raise ValueError("metric must be 'rmse' or 'mae'")

        return result
    else:
        x = np.linspace(y.min(), y.max(), 100)
        yhat_plot = dist.pdf(x)

        plt.hist(y, bins=n_bins, density=True, color="k")
        ax = plt.gca()
        ax.plot(x, yhat_plot)
        txt = "\n".join(np.array(args).round(4).astype(str))
        ax.text(0.95, 0.95, txt, ha="right", va="top", transform=ax.transAxes)

        return ax


def find_best_distribution_fit(data):
    distributions = [
        f
        for f in dir(dist._continuous_distns)
        if (not f.startswith("_") and (f.lower() == f))
    ]
    distributions.remove("levy_stable")
    distributions.remove("studentized_range")

    results = {}
    for d in distributions:
        try:
            func = getattr(dist, d)
            results[d] = fit_distribution(data, dist_func=func, plot=0)
            print("", end=".")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
            pass

    s = pd.Series(results)
    return s


def get_best_distributions_for_df(df):
    results = {}
    for col in df:
        print(col, end="")
        results[col] = find_best_distribution_fit(df[col])
        print()
    return pd.DataFrame(results)
