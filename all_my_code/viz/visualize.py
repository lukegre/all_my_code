
def animate_xda(
    ax, xda, draw_func=None, dim="time", sname="./animation.mp4", fps=6, **plot_kwargs
):
    """
    NOTE: this function is still in development. Would be
    nice to have this as a xarray accessor
    Animate a DataArray along the first dimension.

    Parameters
    ----------
    ax : pyplot.Axes
        The axes object into which you'd like to animate
    xda : xr.DataArray
        the given dimension of the array will be iterated
        over given that the remaining axes is 2D
    draw_func : callable
        function that plots a single time slice of the data
        where a list of changed objects are returned
    dim : str
        the dimension over which the image will be iterated
    sname : str
        name to save the animation to
    fps : int
        frames per second
    plot_kwargs : {}
        keyword pairs that will be passed to the plot function
    """
    from matplotlib import animation, pyplot

    def draw(f):
        print(".", end="")
        return (xda.isel({dim: f}).plot(ax=ax, **plot_kwargs),)

    def animate(frame):
        if draw_func is None:
            return draw(frame)
        elif callable(draw_func):
            return draw_func(frame)

    fig = ax.get_figure()
    nframes = xda[dim].size
    anim = animation.FuncAnimation(
        fig, animate, frames=nframes, blit=True, repeat=False, interval=1000 / fps,
    )

    anim.save(sname, writer=animation.FFMpegWriter(fps=fps))

    pyplot.close(fig)

