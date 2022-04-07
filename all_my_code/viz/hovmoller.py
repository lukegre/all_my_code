def make_zonal_anomaly_plot_data(da):
    da_zon_mean = da.mean('lon').dropna('lat', how='all')
    lat_avg = da_zon_mean.mean('time')
    da_zon_anom = da_zon_mean - lat_avg
    da_zon_anom_an = da_zon_anom.resample(time='1AS').mean().T
    
    return lat_avg, da_zon_anom_an


def plot_zonal_anom(da, **kwargs): 
    from matplotlib import pyplot as plt
    
    fig = plt.figure(figsize=[7.5, 2.5], dpi=140)
    
    ax = [
        plt.subplot2grid([1, 5], [0, 0], colspan=1),
        plt.subplot2grid([1, 5], [0, 1], colspan=4),
    ]
    
    lat_avg, zon_anom = make_zonal_anomaly_plot_data(da)
    
    x1 = lat_avg.values
    y1 = lat_avg.lat.values
    
    ax[0].plot(x1, y1, color='k', lw=1.5)
    
    props = dict(
        cbar_kwargs=dict(pad=0.03),
        robust=True, 
        levels=11, 
        ax=ax[1])
    props.update(kwargs)
    img = zon_anom.plot.contourf(**props)
    
    ax[0].set_ylabel('Latitude (°N)')
    
    plt.sca(ax[1])
    plt.xticks(rotation=0, ha='center')
    ax[1].set_yticks([])
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    
    return ax[0], img