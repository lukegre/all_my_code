===============================
all_my_code
===============================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
        :target: https://www.gnu.org/licenses/gpl-3.0


tools that I've developed over time that form a part of my daily workflow. Thought good to share.
Note that I do not test my code, I do not verify the results. So please treat with caution.
Feel free to drop issues and suggest changes. No guarantee that I'll make the changes though.


Installation
------------
To get the latest version: 

.. code-block:: bash

   pip install git+https://github.com/lukegre/all_my_code/


Usage
-----

Just a few examples of functions that might be useful. This is not a complete list. 
The functions themselves are quite well documented. 


**WARNING: these examples are out of date**

.. code-block:: python

   import all_my_code as amc  # will import xarray methods/accessors too
   from cartopy import crs, feature
   from matplotlib import pyplot as plt
   
   xda = xr.open_dataarray('path_to_demo_data.nc')
   
   # Time series stats #############
   xda.time_series.climatology(tile=True)
   xda.time_series.detrend(dim='time', deg=2)
   xda.time_series.deseasonalise()
   
   # PLOTTING ############
   # maps with xarray
   img = xda.isel(time=0).map()
   img.colorbar.set_label('some label')
   
