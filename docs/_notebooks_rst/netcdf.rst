.. This file was generated automatically from the ipython notebook:
.. notebooks/netcdf.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_netcdf:


NetCDF reading and writing
==========================

.. _Read_from_one_netCDF_file:

Read from one netCDF file
-------------------------

>>> from dimarray import read_nc, get_datadir
>>> import os
>>> ncfile = os.path.join(get_datadir(), 'cmip5.CSIRO-Mk3-6-0.nc')  # get one netCDF file
>>> data = read_nc(ncfile)  # load full file
>>> data
Dataset of 2 variables
dimensions: 'time', 'scenario'
0 / time (451): 1850 to 2300
1 / scenario (5): historical to rcp85
tsl: ('time', 'scenario')
temp: ('time', 'scenario')

Then access the variable of choice

>>> %matplotlib inline # doctest: +SKIP 
>>> data['temp'].plot() # doctest: +SKIP
<matplotlib.axes.AxesSubplot at 0x7fd3ca039610>

.. image:: netcdf_files/figure_4-1.png



Load only one variable

>>> data = read_nc(ncfile,'temp') # only one variable
>>> data = read_nc(ncfile,'temp', indices={"time":slice(2000,2100), "scenario":"rcp45"})  # load only a chunck of the data
>>> data = read_nc(ncfile,'temp', indices={"time":1950.3}, tol=0.5)  #  approximate matching, adjust tolerance
>>> data = read_nc(ncfile,'temp', indices={"time":-1}, indexing='position')  #  integer position indexing


.. __Read_from_multiple_files:

 Read from multiple files
-------------------------

Read variable 'temp' across multiple files (representing various climate models). 
In this case the variable is a time series, whose length may vary across experiments 
(thus align=True is passed to reindex axes before stacking). Under the hood the function 
py:func:`dimarray.stack` is called:

>>> direc = get_datadir()
>>> temp = read_nc(direc+'/cmip5.*.nc', 'temp', align=True, axis='model')


A new 'model' axis is created labeled with file names. It is then 
possible to rename it more appropriately, e.g. keeping only the part
directly relevant to identify the experiment:

>>> getmodel = lambda x: os.path.basename(x).split('.')[1] # extract model name from path
>>> temp.reset_axis(getmodel, axis='model', inplace=True) # would return a copy if inplace is not specified
>>> temp
dimarray: 9114 non-null elements (6671 null)
dimensions: 'model', 'time', 'scenario'
0 / model (7): IPSL-CM5A-LR to CSIRO-Mk3-6-0
1 / time (451): 1850 to 2300
2 / scenario (5): historical to rcp85
array(...)

This works on datasets as well

>>> ds = read_nc(direc+'/cmip5.*.nc', align=True, axis='model')
>>> ds.reset_axis(getmodel, axis='model', inplace=True)
>>> ds
Dataset of 2 variables
dimensions: 'model', 'time', 'scenario'
0 / model (7): IPSL-CM5A-LR to CSIRO-Mk3-6-0
1 / time (451): 1850 to 2300
2 / scenario (5): historical to rcp85
tsl: ('model', 'time', 'scenario')
temp: ('model', 'time', 'scenario')

.. _Write_to_netCDF_:

Write to netCDF 
----------------

Let's define some dummy arrays representing temperature in northern and southern hemisphere for three years.

>>> from dimarray import DimArray
>>> temperature = DimArray([[1.,2,3], [4,5,6]], axes=[['north','south'], [1951, 1952, 1953]], dims=['lat', 'time'])
>>> global_mean = temperature.mean(axis='lat')  
>>> climatology = temperature.mean(axis='time')


Let's define a new dataset

>>> from dimarray import Dataset
>>> ds = Dataset({'temperature':temperature, 'global':global_mean})
>>> ds
Dataset of 2 variables
dimensions: 'time', 'lat'
0 / time (3): 1951 to 1953
1 / lat (2): north to south
global: ('time',)
temperature: ('lat', 'time')

Saving the dataset to file is pretty simple:

>>> ds.write_nc('/tmp/test.nc', mode='w')


It is possible to append more variables

>>> climatology.write_nc('/tmp/test.nc', 'climatology')  # by default mode='a+'


Just as a check, all three variables seem to be there:

>>> read_nc('/tmp/test.nc')
Dataset of 3 variables
dimensions: 'time', 'lat'
0 / time (3): 1951 to 1953
1 / lat (2): north to south
global: ('time',)
climatology: ('lat',)
temperature: ('lat', 'time')

Note that when appending a variable to a netCDF file or to a dataset, its axes must match, otherwise an error will be raised. In that case it may be necessary to reindex an axis (see :ref:`page_reindexing`). When initializing a dataset with bunch of dimarray however, reindexing is performed automatically.