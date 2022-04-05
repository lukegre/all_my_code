from matplotlib import pyplot as plt


def plot_ensemble_line_with_std(da, x='time', ax=None, **lineplot_kwargs):
    from seaborn import lineplot
    
    if ax is None:
        ax = plt.gca()
        
    if not getattr(da, 'name', False):
        name = 'data'
        da = da.rename(name)
    else:
        name = da.name
    
    df = da.to_dataframe(name=name).reset_index()
    
    props = dict(ci='sd')
    props.update(lineplot_kwargs)
    line = lineplot(data=df, x=x, y=name, **props)
    return line
    


def plot_time_series(xda, ax=None, label_spacing=0.02, **kwargs):
    """
    Plots time on x-axis and other var on y-axis. 
    
    Args:
        xda (xr.DataArray): must have time as one of the dimensions, lines will 
            be plotted and labelled with the other dimension. 
        ax (plt.Axes): an axes object that will be drawn into
        label_spacing (float): labels are drawn at the end of the lines. The 
            spacing determines how far to the right of the line the label will 
            be located. A fraction of the total time series length. 
            
    Returns:
        plt.Figure: figure object
        plt.Axes: axes object
    """
    from matplotlib import pyplot as plt
    import numpy as np
    
    assert 'time' in xda.dims, 'time must be in data array'
    
    if ax is None:
        fig, ax = plt.subplots(figsize=[6, 2.4])
    else:
        fig = ax.get_figure()
    
    key = list(set(xda.dims) - set(['time']))[0]
    line = xda.plot(hue=key, ax=ax, add_legend=False, **kwargs)

    t = xda.time.values
    labels_x_pos = t[-1] + (t[-1] - t[0]) * label_spacing
    
    last_y_value = [l.get_data()[1][-1] for l in ax.get_lines()]
    colors = [l.get_color() for l in ax.get_lines()]
    labels = xda[key].values.tolist()

    props = dict(va='center', weight='bold')
    ax.labels = []
    for i in range(len(labels)):
        ax.labels += ax.text(
            labels_x_pos, 
            last_y_value[i], 
            labels[i], 
            c=f'C{i}', 
            **props),
    
    ax.set_title('')
    ax.set_xlabel('')
    plt.xticks(rotation=0, ha='center')
    
    return fig, ax


def style_line_subplot(ax, add_zero_line=True, xlim=None, y_range=None):
    import numpy as np
    
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    
    plt.xticks(rotation=0, ha='center')
    plt.xlabel('')
    plt.title('')
    
    if (ax.get_ylim()[0] < 0 < ax.get_ylim()[1]) & add_zero_line:
        ax.axhline(0, color='k', ls='--', lw=0.5, zorder=0)
    
    if xlim is None:
        xlim = ax.get_xlim()
    ax.set_xticks(np.arange('1980', '2020', 5, dtype='datetime64[Y]'))
    ax.set_xticklabels(np.arange(1980, 2020, 5))
    ax.set_xlim(*xlim)
    
    if y_range is not None:
        center = np.mean(ax.get_ylim())
        upp = center + y_range/2
        low = center - y_range/2
        ax.set_ylim(low, upp)
        
    return ax


def get_parent_axes_from_line_obj(line):
    """
    Will get the axes from a given line object
    """
    fig = line.get_figure()
    subplots = fig.get_axes()
    
    for ax in subplots:
        if line in ax.get_lines():
            return ax

        
def annotate_line(line, xloc, label=None, **kwargs):
    """
    takes the line object and it's label. Places that label
    next to the line at the given x-location. Means that 
    y-loc does not have to be specified. 
    """
    import numpy as np
    from ..munging.date_utils import convert_datestring_to_datetime

    if label is None:
        label = line.get_label()
        
    if isinstance(xloc, str):
        xloc = convert_datestring_to_datetime(xloc)
    
    fig = line.get_figure()
    ax = get_parent_axes_from_line_obj(line)
    
    x, y = line.get_data()
    xdif = abs(x - xloc)
    
    i = np.nanargmin(xdif)
    x = x[i]
    y = y[i]
    
    lw = line.get_lw()
    fw = fig.get_figwidth()
    aw = ax.get_position().width
    
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()
    
    text = ax.text(x, y, label, **kwargs)
    
    return text