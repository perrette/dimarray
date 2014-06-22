dimarray
========

pandas in N dimensions
----------------------
dimarray is a package to handle numpy arrays with labelled dimensions and axes. 
Inspired from pandas, it includes advanced alignment and reshaping features and 
as well as nan handling.

The main difference with pandas is that it is generalized to N dimensions, and behaves more closely to a numpy array. 
The axes do not have fixed names ('index', 'columns', etc...) but are 
given a meaningful name by the user (e.g. 'time', 'items', 'lon' ...). 
This is especially useful for high dimensional problems such as sensitivity analyses.

A natural I/O format for such an array is netCDF, common in geophysics, which rely on 
the netCDF4 package, and supports metadata.


Getting started
----------------

A **``DimArray``** can be defined just like a numpy array, with
additional information about its dimensions, which can be provided
via its `axes` and `dims` parameters:

>>> from dimarray import DimArray
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

Or can just be done **a la numpy**, via integer index:

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

Documentation:
--------------
See more on:
https://pythonhosted.org/dimarray/


Download latest version on GitHub:
----------------------------------
https://github.com/perrette/dimarray

Installation:
-------------

Requires

- python2.7
- numpy
- netCDF4 (optional) :  for netCDF I/O
  
    repository: https://github.com/Unidata/netcdf4-python

    From source:
        the documentation can be found there: http://unidata.github.io/netcdf4-python/
        Basically you need to install HDF5 and netCDF4 libraries on your system before
        using pip or your favorite package manager.
    
    This can be annoying to install HDF5 and netCDF4 from source.
    Using the anaconda package from continuum analytics save time 
    (That is the only one I tried, but it possibly also 
    works with Enthought, xyPython or some other pre-compiled version of python)
    With conda (the package manager shipped with - but kind of independent from - anaconda) 
    it is enough to do a simple:

        conda install netCDF4 

- matplotlib (optional) : for plotting (for now plot command also requires pandas)
- pandas (optional) :  to_pandas() and from_pandas() methods, plot()

sudo python setup.py install

or

pip install dimarray
