===============================
all_my_code
===============================


.. image:: https://img.shields.io/travis/luke-gregor/OceanDataTools.svg
        :target: https://travis-ci.org/luke-gregor/OceanDataTools
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
        :target: https://www.gnu.org/licenses/gpl-3.0


tools that I've developed over time that form a part of my daily workflow. Thought good to share.


Installation
------------
To get the latest version: 

.. code-block:: bash

   pip install git+git://github.com/lukegre/all_my_code/


Usage
-----

Just a few examples of functions that might be useful. This is not a complete list. 
The functions themselves are quite well documented. 


**WARNING: these examples are out of date**

.. code-block:: python

   import ocean_data_tools as odt  # will import xarray methods/accessors too
   from cartopy import crs, feature
   from matplotlib import pyplot as plt
   
   xda = xr.open_dataarray('path_to_demo_data.nc')
   
   # STATS #############
   xda.stats.trend()
   xda.stats.detrend()
   xda.stats.pca_decomp()
   
   # PLOTTING ############
   # maps with xarray
   ax2 = xda.isel(time=0).plot_map()
   fig1 = ax1.get_figure()
   
   # nice style for time series (still under development)
   ts_data = xda.mean(dim=['lat', 'lon'])
   ax2 = ts_data.plot()
   fig2 = ax2.get_figure()
   odt.plotting.pimp_plot(ax2)
   
   # save a list of figures to a single PDF with a figure per page
   odt.plotting.figs_to_pdf([fig1, fig2])
   
