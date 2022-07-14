def test_seas_cycle_fit(method="clim", ax=None):
    import matplotlib.pyplot as plt
    import xarray as xr
    import pandas as pd
    import numpy as np

    plt.rcParams["mathtext.fontset"] = "cm"

    if ax is None:
        fig, ax = plt.subplots(dpi=120, figsize=(6, 3))

    # create our dummy array
    time = pd.date_range("1990", "2020", freq="1MS", closed="left")
    x = np.linspace(0, 30 * 2 * np.pi, 12 * 30)
    dummy_array = (
        xr.DataArray(
            data=np.sin(x) * x,
            dims=["time"],
            coords={"time": time},
        )
        .shift(time=-2)
        .rename("Toy data ($\sin(x) \\times x$)")
    )

    if method == "clim":
        jja_minus_djf = (
            dummy_array.stats.seascycl_fit_climatology().jja_minus_djf.rename(
                r"∆$^{\rm{seas}}$ Climatology"
            )
        )
    elif method == "graven":
        jja_minus_djf = (
            dummy_array.expand_dims(lat=[1], lon=[1])
            .stats.seascycl_fit_graven()
            .jja_minus_djf.squeeze(drop=True)
            .rename(r"∆$^{\rm{seas}}$ Graven (2013)")
        )
    else:
        raise ValueError(f"{method} is not a valid method (clim/graven")

    amplitude = jja_minus_djf / 2

    dummy_array.plot(lw=1, ax=ax, color="k", label=dummy_array.name)
    (line,) = (+amplitude).plot(lw=3, ax=ax, alpha=0.5, label=jja_minus_djf.name)
    (-amplitude).plot(color=line.get_c(), lw=line.get_lw(), ax=ax, alpha=0.5)
    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.legend(loc=0)
    ax.set_xlabel("")

    return ax
