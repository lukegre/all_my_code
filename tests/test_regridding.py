def test_grid_flat_like_xarray():
    from all_my_code.munging.collocation import grid_flat_dataframe_to_target
    import pandas as pd
    import xarray as xr
    import numpy as np

    target_grid = xr.DataArray(
        np.ndarray([12, 90, 180]),
        dims=("time", "lat", "lon"),
        coords=dict(
            time=pd.date_range("2000-01", periods=12, freq="MS") + pd.Timedelta("14D"),
            lat=np.arange(-89, 90.0, 2),
            lon=np.arange(-179, 180.0, 2),
        ),
    )

    n = 500
    t0 = np.datetime64("2000-01-01")
    flat = pd.DataFrame(
        np.array(
            [
                np.random.normal(0, 45, n),
                np.random.uniform(-180, 180, n),
                np.random.uniform(-100, 100, n),
            ]
        ).T,
        columns=["lat", "lon", "vals"],
    ).astype(float)

    flat["time"] = t0 + np.random.randint(0, 365, n).astype("timedelta64[D]")

    grid_flat_dataframe_to_target(flat, target_grid)
