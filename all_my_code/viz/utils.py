def label_subplots(axes_list, labels=None, loc="upper left", lw=0, **kwargs):
    """
    Labels subplots in a list of axes.

    Parameters
    ----------
    axes_list : list
        list of axes to label
    labels : list   [optional]
        list of labels to use. If not provided, will use the alphabet.
    loc : str
        location of labels same as ``plt.legend``
    lw : int
        line width of the box around the labels
    kwargs : key-value pairs
        passed to plt.AnchoredText
        For a white box behind the number use
        prop=dict(backgroundcolor="w", size=12, weight="bold")


    Returns
    -------
    Artists:
        list of artists
    """
    from string import ascii_lowercase
    from matplotlib.offsetbox import AnchoredText

    if labels is None:
        labels = ascii_lowercase

    if lw == 0:
        frameon = False
    else:
        frameon = True

    props = dict(
        prop=dict(size=12, weight="bold"),
        loc=loc,
        frameon=frameon,
        pad=0.4,
        borderpad=0,
    )
    props.update(kwargs)

    height = props.pop("height", 0.5)
    width = props.pop("width", height)  # noqa: F841

    artists = []
    for ax, lbl in zip(axes_list, labels):
        at = AnchoredText(lbl, **props)
        at.patch.set_lw(lw)
        artists += (ax.add_artist(at),)

    return artists


def get_line_from_label(ax, label_contains):
    """
    Get a line object that has a label that contains the string

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to search
    label_contains : str
        The string to search for in the line labels

    Returns
    -------
    line : matplotlib.lines.Line2D
        The line object that has the label or None if not found
    """

    lines = ax.get_lines()
    for line in lines:
        label = line.get_label()
        if label_contains in label:
            return line
    return None


def save_figures_to_pdf(fig_list, pdf_name, return_figures=False, **savefig_kwargs):
    """
    Saves a list of figure objects to a pdf with multiple pages.
    Parameters
    ----------
    fig_list : list
        list of figure objects
    pdf_name : str
        path to save pdf to.
    savefig_kwargs : key-value pairs passed to ``Figure.savefig``
    Returns
    -------
    None
    """
    import matplotlib.backends.backend_pdf
    from matplotlib import pyplot as plt

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

    kwargs = dict(dpi=120)
    kwargs.update(savefig_kwargs)

    for fig in fig_list:  # will open an empty extra figure :(
        pdf.savefig(fig, **kwargs)
    pdf.close()

    if return_figures:
        return fig_list
    else:
        plt.close("all")


def add_colorbar_cumsum_count_from_map(img, cb=None, ha="left"):
    """
    Adds the percentage of the data that falls in a given
    level for a discrete color map.

    Parameters
    ----------
    img : matplotlib.image.AxesImage
        The image that contains the levels - must contain
        'allsegs' attribute (matplotlib.contour[f])
    cb : matplotlib.colorbar.Colorbar
        The colorbar to add the percentage to. If not provided,
        then will look in img for 'colorbar' attribute.

    Returns
    -------
    cb : matplotlib.colorbar.Colorbar
        The colorbar with the percentage added.
    """

    import pandas as pd
    import numpy as np
    import matplotlib as mpl

    raise NotImplementedError("the function doesn't work at the moment")

    def get_cumsum_count_from_contourf(img):
        counts = {}
        for level, segment in zip(img.levels, img.allsegs):
            counts[level] = sum([len(seg) for seg in segment])

        df = pd.Series(counts).to_frame(name="counts")
        df["count_cumsum"] = df.counts.cumsum()
        df["percent"] = df.counts / df.counts.sum() * 100
        df["pct_cumsum"] = df.percent.cumsum()
        return df

    def get_cumsum_count_from_path_collection(path_collection):
        y = path_collection.get_array()
        # we look for the quadmesh object
        for child in cb.ax.get_children():
            if isinstance(child, mpl.collections.QuadMesh):
                bin_centers = child.get_array()
                break
            else:
                bin_centers = None

        if bin_centers is None:
            raise ValueError("Could not find quadmesh object in colorbar")

        if len(bin_centers) > 15:
            raise ValueError(
                f"There are too many colors ({bin_centers}) - make it more discrete"
            )

        # now we make the bin edges
        db = float(np.diff(bin_centers).mean())  # we need the delta bins
        b0, b1 = bin_centers.min(), bin_centers.max()
        bin_edges = np.arange(b0 - (db / 2), b1 + db, db)

        df = pd.DataFrame()
        df["counts"] = np.histogram(y, bin_edges)[0]
        df["percent"] = np.histogram(y, bin_edges, density=True)[0] * 100
        df["count_cumsum"] = df.counts.cumsum()
        df["pct_cumsum"] = df.percent.cumsum()
        return df

    # cb is used by get_cumsum_count_from_path_collection so must run first
    if cb is None:
        if hasattr(img, "colorbar"):
            cb = img.colorbar
        else:
            raise KeyError("No 'colorbar' found in img, please provide cb")

    if hasattr(img, "allsegs"):
        df = get_cumsum_count_from_contourf(img)
    elif isinstance(img, mpl.collections.PathCollection):
        df = get_cumsum_count_from_path_collection(img)
    else:
        raise TypeError(
            "only QuadMesh and PathCollection are supported inputs for 'img'"
        )

    x = np.convolve(df.index.values, [0.5, 0.5], mode="valid")
    dx = np.diff(x).mean()
    df["x"] = np.r_[x, x[-1] + dx]

    print(df)
    if cb.extend == "min":
        df = df.iloc[1:]
    elif cb.extend == "max":
        df = df.iloc[:-1]
    elif cb.extend == "both":
        df = df.iloc[1:-1]

    xlim = cb.ax.get_ylim()
    half = (xlim[1] - xlim[0]) / 2
    cb.percentages = []
    for key in df.index:
        color = "w" if key < half else "k"
        x = df.loc[key, "x"]
        s = df.loc[key, "percent"]
        s = round(s, 1) if s < 10 else round(s)
        props = dict(ha="center", va="center", alpha=0.7, zorder=10)
        if cb.orientation.startswith("v"):
            y, x = x, half
            props.update(rotation=90)
        else:
            y = half
        cb.percentages += (cb.ax.text(x, y, f"{s}%", color=color, **props),)

    if ha == "left":
        props = dict(x=-0.01, y=0.5, ha="right", va="center")
    elif ha == "right":
        props = dict(x=1.01, y=0.5, ha="left", va="center")
    cb.ax.text(**props, s="Distribution\nof data", transform=cb.ax.transAxes, alpha=0.7)

    return cb
