from matplotlib import pyplot as plt


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
    import pandas as pd
    
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
