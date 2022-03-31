def add_subplot_square(ax, w=0.05, h=None, loc='upper left', **bbox_props):
    def get_aspect(ax):
        from operator import sub
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # Total figure size
        figW, figH = ax.get_figure().get_size_inches()
        # Axis size on figure
        _, _, w, h = make_axes_locatable(ax).get_position()

        # Ratio of display units
        disp_ratio = (figH * h) / (figW * w)
        # Ratio of data units
        # Negative over negative because of the order of subtraction
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

        return disp_ratio
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    aspect = get_aspect(ax)
    
    if h is None:
        h = w / aspect 
    else: 
        h = h
        
    if loc == 'lower left':
        y0 = 0
        x0 = 0
    elif loc == 'upper left':
        y0 = 1 - h
        x0 = 0
    elif loc == 'lower right':
        x0 = 1 - w
        y0 = 0
    elif loc == 'upper right':
        x0 = 1 - w
        y0 = 1 - h
    else:

        raise KeyError(
            'loc must be one of the following combinations: '
            'lower/upper, left/right')
    
    props = dict(
        linewidth=0, 
        edgecolor='none', 
        facecolor='k', 
        transform=ax.transAxes,
        zorder=10,
    )
    props.update(bbox_props)
    rect = patches.Rectangle(
        (x0, y0), w, h, **props)

    ax.add_patch(rect)
    
    return x0 + w/2, y0 + h/2


def corner_text(ax, text, square_width=0.05, loc='upper left', bbox_props={'facecolor': 'k'}, **text_props):
    x, y = add_subplot_square(ax, w=square_width, loc=loc, **bbox_props)
    
    props = dict(
        color='w' if bbox_props.get('facecolor', 'k') == 'k' else 'w',
        size=11,
        weight='bold',
        zorder=11,
        ha='center',
        va='center',
        transform=ax.transAxes,
    )
    props.update(text_props)
    
    ax.number = ax.text(x, y, text, **props)
    
    return ax


def save_figures_to_pdf(fig_list, pdf_name, **savefig_kwargs):
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
    plt.close("all")
    
