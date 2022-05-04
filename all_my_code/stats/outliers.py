def mask_outliers_iqr(da, dim=None, iqr_mult=3):
    q1, q3, iqr = _get_iqr(da, dim=dim)

    lower = q1 - iqr * iqr_mult
    upper = q3 + iqr * iqr_mult

    filtered = da.where(lambda x: (x > lower) & (x < upper))
    return filtered


def mask_outliers_std(da, dim=None, std_mult=3):
    std = da.std(dim=dim)
    avg = da.mean(dim=dim)
    lower = avg - std * std_mult
    upper = avg + std * std_mult
    filtered = da.where(lambda x: (x > lower) & (x < upper))
    return filtered


def _get_iqr(da, dim=None):
    q = da.quantile([0.25, 0.75], dim=dim)
    q1 = q.sel(quantile=0.25)
    q3 = q.sel(quantile=0.75)
    iqr = q3 - q1
    return q1, q3, iqr
