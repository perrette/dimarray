Introduction
============

.. image:: https://travis-ci.org/perrette/dimarray.svg?branch=master
   :target: https://travis-ci.org/perrette/dimarray

Numpy array with dimensions
---------------------------
dimarray is a package to handle numpy arrays with labelled dimensions and axes. 
Inspired from pandas, it includes advanced alignment and reshaping features and 
as well as missing-value (NaN) handling.

The main difference with pandas is that it is generalized to N dimensions, and behaves more closely to a numpy array. 
The axes do not have fixed names ('index', 'columns', etc...) but are 
given a meaningful name by the user (e.g. 'time', 'items', 'lon' ...). 
This is especially useful for high dimensional problems such as sensitivity analyses.

A natural I/O format for such an array is netCDF, common in geophysics, which relies on 
the netCDF4 package, and supports metadata.


License
-------
dimarray is distributed under a 3-clause ("Simplified" or "New") BSD
license. Parts of basemap which have BSD compatible licenses are included.
See the LICENSE file, which is distributed with the dimarray package, for details.

Getting started
---------------

A **``DimArray``** can be defined just like a numpy array, with
additional information about its dimensions, which can be provided
via its `axes` and `dims` parameters:

>>> from dimarray import DimArray
>>> a = DimArray([[1.,2,3], [4,5,6]], axes=[['a', 'b'], [1950, 1960, 1970]], dims=['variable', 'time']) 
>>> a
dimarray: 6 non-null elements (0 null)
0 / variable (2): 'a' to 'b'
1 / time (3): 1950 to 1970
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])

Indexing now works on axes

>>> a['b', 1970]
6.0

Or can just be done **a la numpy**, via integer index:

>>> a.ix[0, -1]
3.0

Basic numpy transformations are also in there:

>>> a.mean(axis='time')
dimarray: 2 non-null elements (0 null)
0 / variable (2): 'a' to 'b'
array([ 2.,  5.])

Can export to `pandas` for pretty printing:

>>> a.to_pandas()
time      1950  1960  1970
variable                  
a            1     2     3
b            4     5     6

.. _links:

Useful links
------------
================================    ====================================
Documentation                       http://dimarray.readthedocs.org
Code on github (bleeding edge)      https://github.com/perrette/dimarray
Code on pypi   (releases)           https://pypi.python.org/pypi/dimarray
Mailing List                        http://groups.google.com/group/dimarray
Issues Tracker                      https://github.com/perrette/dimarray/issues
================================    ====================================

Install
-------

**Requirements**:

- python 2.7   
- numpy (tested with 1.7, 1.8, 1.9, 1.10.1)

**Optional**:

- netCDF4 (tested with 1.0.8, 1.2.1) (netCDF archiving) (see notes below)
- matplotlib 1.1 (plotting)
- pandas 0.11 (interface with pandas)
- cartopy 0.11 (dimarray.geo.crs)

Download the latest version from github and extract from archive
Then from the dimarray repository type (possibly preceded by sudo):

.. code:: bash
    
    python setup.py install  

Alternatively, you can use pip to download and install the version from pypi (could be slightly out-of-date):

.. code:: bash

    pip install dimarray 


Notes on installing netCDF4
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- On Ubuntu, using apt-get is the easiest way (as indicated at https://github.com/Unidata/netcdf4-python/blob/master/.travis.yml):


.. code:: bash

   sudo apt-get install libhdf5-serial-dev netcdf-bin libnetcdf-dev

- On windows binaries are available: http://www.unidata.ucar.edu/software/netcdf/docs/winbin.html

- From source. Installing the netCDF4 python module from source can be cumbersome, because 
it depends on netCDF4 and (especially) HDF5 C libraries that need to 
be compiled with specific flags (http://unidata.github.io/netcdf4-python). 
Detailled information on Ubuntu: https://code.google.com/p/netcdf4-python/wiki/UbuntuInstall

Contributions
-------------
All suggestions for improvement or direct contributions are very welcome.
You can ask a question or start a discussion on the mailing list
or open an `issue` on github for precise requests. See `links`_.
