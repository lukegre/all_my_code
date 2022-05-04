import numpy as np
import xarray as xr


def get_woa18(name, month=np.arange(1, 13), save_dir="~/Data/cached/", verbose=True):
    """
    Get/download the WOA18 data for a given variable.

    Parameters
    ----------
    name : str
        The name of the variable to get: silicate, phosphate, nitrate,
        oxygen, AOU (apparent oxygen utilization).
    month : array_like, optional
        The months to get data for. Default is all months.
    save_dir : str, optional
        The directory to save the data to. Default is ~/Data/cached/.

    Returns
    -------
    xarray.Dataset
        World Ocean Atlas data for the given variable
    """
    from pathlib import Path

    if verbose:
        vprint = print
    else:
        vprint = lambda *a, **k: None

    if not isinstance(name, str):
        props = dict(month=month, save_dir=save_dir, verbose=verbose)
        ds_list = [get_woa18(n, **props) for n in name]
        return xr.merge(ds_list)

    options = dict(silicate="i", phosphate="p", nitrate="n", oxygen="o", AOU="A")

    assert name in options, f"'name' must be one of {list(options)}"

    abbrev = options[name]
    key = f"{abbrev}_an"
    url = (
        "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/"
        f"woa/{name}/all/1.00/woa18_all_{abbrev}{{:02d}}_01.nc"
    )

    save_dir = save_dir.format(name=name, abbrev=abbrev)
    sname = str(Path(save_dir).expanduser() / Path(url).name)

    ds_list = []
    vprint("loading:", sname)
    for m in month:
        fname = sname.format(m)
        if Path(fname).is_file():
            ds_list += (xr.open_dataset(fname, chunks={"month": 1}),)

        else:
            vprint(m, end=" ")
            ds = (
                xr.open_dataset(url.format(m), decode_times=False)[[key]]
                .assign_coords(time=[m])
                .rename(**{"time": "month", key: name})
                .chunk(dict(month=1))
            )
            vprint("downloading and saving to: ", sname.format(m))
            ds.to_netcdf_with_compression(sname.format(m))
            ds_list += (ds,)

    ds = xr.concat(ds_list, "month")
    return ds
