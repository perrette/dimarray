dimarray documentation
======================

Concept:
--------
A `numpy` array with labelled axes and dimensions, in the spirit of 
`pandas` but not restricted to low-dimensional arrays, with improved
indexing, and using the full power of having dimension names 
(see below "Comparisons with other packages").

Having axis name and axis values allow on-the-fly axis alignment and 
dimension broadcasting in basic operations (addition, etc...), 
so that rules can be defined for nearly every sequence of operands. 

A natural I/O format for such an array is netCDF, common in geophysics, which rely on 
the netCDF4 package, and supports metadata

Documentation:
--------------
Check out  http://dimarray.readthedocs.org

Download latest version on GitHub:
----------------------------------
https://github.com/perrette/dimarray

Requires:
---------
- numpy

- netCDF4 (optional) :  for netCDF I/O, found at https://code.google.com/p/netcdf4-python/)

- matplotlib (optional) : for plotting (for now plot command also requires pandas)

- pandas (optional) :  to_pandas() and from_pandas() methods, plot()

Installation:
-------------
sudo python setup.py install

Getting started
----------------

A **``DimArray``** can be defined just like a numpy array, with
additional information about its dimensions, which can be provided
via its `axes` and `dims` parameters:

>>> from dimarray import DimArray, Dataset
>>> a = DimArray([[1.,2,3], [4,5,6]], axes=[['a', 'b'], [1950, 1960, 1970]], dims=['variable', 'time']) 
>>> a
dimarray: 6 non-null elements (0 null)
dimensions: 'variable', 'time'
0 / variable (2): a to b
1 / time (3): 1950 to 1970
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])

Indexing now works on axes
>>> a['b', 1970]
6.0

Or can just be done _a la numpy_, via integer index:
>>> a.ix[0, -1]
3.0

Basic numpy transformations are also in there:
>>> a.mean(axis='time')
dimarray: 2 non-null elements (0 null)
dimensions: 'variable'
0 / variable (2): a to b
array([ 2.,  5.])

But never stays very far from `pandas`:
>>> a.to_pandas()
time      1950  1960  1970
variable                  
a            1     2     3
b            4     5     6


More in the [Getting Started](http://dimarray.readthedocs.org/en/latest/_build_rst/getting_started.html) section of the [doc](http://dimarray.readthedocs.org)
