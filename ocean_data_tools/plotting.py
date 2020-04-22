
def pimp_plot(ax, **grid_kwargs):

    ax.set_xlabel('')

    grd_kwds = dict(
        lw=0.5, color='lightgrey'
    )
    grd_kwds.update(grid_kwargs)

    ax.hlines(ax.get_yticks(), *ax.get_xlim(), **grd_kwds)

    ax.xaxis.set_tick_params(color='#aaaaaa', which='both')
    ax.yaxis.set_tick_params(color='#aaaaaa', which='both')
    [ax.spines[s].set_visible(False) for s in ['right', 'top']]
    [ax.spines[s].set_color('#CCCCCC') for s in ax.spines]
    [ax.spines[s].set_linewidth(1) for s in ax.spines]

    return ax
