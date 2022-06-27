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


def add_colorbar_cumsum_count_from_map(img, cb=None):
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

    def get_cumsum_count_from_contourf(img):
        counts = {}
        for level, segment in zip(img.levels, img.allsegs):
            counts[level] = sum([len(seg) for seg in segment])

        df = pd.Series(counts).to_frame(name="counts")
        df["count_cumsum"] = df.counts.cumsum()
        df["percent"] = df.counts / df.counts.sum() * 100
        df["pct_cumsum"] = df.percent.cumsum()
        return df

    def get_cumsum_count_from_scatter(img):
        msg = "Cannot get cumsum count from scatter plot yet"
        raise NotImplementedError(msg)

    if hasattr(img, "allsegs"):
        df = get_cumsum_count_from_contourf(img)
    else:
        raise TypeError("Only works for contourf at the moment")

    if cb is None:
        if hasattr(img, "colorbar"):
            cb = img.colorbar
        else:
            raise KeyError("No 'colorbar' found in img, please provide cb")

    x = np.convolve(df.index.values, [0.5, 0.5], mode="valid")
    dx = np.diff(x).mean()
    df["x"] = np.r_[x, x[-1] + dx]

    if cb.extend == "min":
        df = df.iloc[1:]
    elif cb.extend == "max":
        df = df.iloc[:-1]
    elif cb.extend == "both":
        df = df.iloc[1:-1]

    half = df.x.max() / 2
    cb.percentages = []
    for key in df.index:
        color = "w" if key < half else "k"
        x = df.loc[key, "x"]
        s = df.loc[key, "percent"]
        props = dict(ha="center", va="center", alpha=0.5)
        cb.percentages += (cb.ax.text(x, half, f"{s:.0f}%", color=color, **props),)

    cb.ax.text(
        -0.01,
        0.5,
        "Percentage\nof data",
        ha="right",
        va="center",
        transform=cb.ax.transAxes,
        alpha=0.5,
    )

    return cb
