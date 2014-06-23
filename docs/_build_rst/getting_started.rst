.. This file was generated automatically from the ipython notebook:
.. notebooks/getting_started.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

..  _page_getting_started:


Tutorial
========

..  _define_a_dimarray:

define a dimarray
-----------------

A **``DimArray``** can be defined just like a numpy array, with
additional information about its axes, which can be given
via `axes` and `dims` parameters.

>>> from dimarray import DimArray, Dataset
>>> a = DimArray([[1.,2,3], [4,5,6]], axes=[['a', 'b'], [1950, 1960, 1970]], dims=['variable', 'time'])
>>> a
dimarray: 6 non-null elements (0 null)
dimensions: 'variable', 'time'
0 / variable (2): a to b
1 / time (3): 1950 to 1970
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])

..  _data_structure:

data structure
--------------

Array data are stored in a `values` **attribute**:

>>> a.values
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])

while its axes are stored in `axes`

>>> a.axes
dimensions: 'variable', 'time'
0 / variable (2): a to b
1 / time (3): 1950 to 1970

For more information refer to section on :ref:`page_data_structure` (as well as :py:class:`dimarray.Axis` and :py:class:`dimarray.Axes`)

..  _numpy-like_attributes:

numpy-like attributes
---------------------

Numpy-like attributes `dtype`, `shape`, `size` or `ndim` are defined, and are not augmented with `dims` and `labels`

>>> a.shape
(2, 3)

>>> a.dims      # grab axis names (the dimensions)
('variable', 'time')

>>> a.labels   # grab axis values
(array(['a', 'b'], dtype=object), array([1950, 1960, 1970]))

..  _indexing_:

indexing 
---------

**Indexing** works on labels just as expected, including `slice` and boolean array.

>>> a['b', 1970]
6.0

but integer-index is always possible via `ix` toogle between `labels`- and `position`-based indexing:

>>> a.ix[0, -1]
3.0

See also: documentation on :ref:`page_indexing`

..  _transformation:

transformation
--------------

Standard numpy transformations are defined, and now accept axis name:

>>> a.mean(axis='time')
dimarray: 2 non-null elements (0 null)
dimensions: 'variable'
0 / variable (2): a to b
array([ 2.,  5.])

and can ignore **missing values (nans)** if asked to:

>>> import numpy as np 
>>> a['a',1950] = np.nan
>>> a.mean(axis='time', skipna=True)
dimarray: 2 non-null elements (0 null)
dimensions: 'variable'
0 / variable (2): a to b
array([ 2.5,  5. ])

See also: documentation on :ref:`Along-axis_transformations`

..  _data_alignment__automatic_broadcasting_and_reindexing:

data alignment: automatic broadcasting and reindexing
-----------------------------------------------------

During an operation, arrays are **automatically re-indexed** to span the 
same axis domain, with nan filling if needed. 
This is quite useful when working with partly-overlapping time series or 
with incomplete sets of items.

>>> yearly_data = DimArray([0, 1, 2], axes=[[1950, 1960, 1970]], dims=['year'])  
>>> incomplete_yearly_data = DimArray([10, 100], axes=[[1950, 1960]], dims=['year']) # last year 1970 is missing
>>> yearly_data + incomplete_yearly_data
dimarray: 2 non-null elements (1 null)
dimensions: 'year'
0 / year (3): 1950 to 1970
array([  10.,  101.,   nan])

A check is also performed on the dimensions, to ensure consistency of the data.
If dimensions do not match this is not interpreted as an error but rather as a 
combination of dimensions. For example, you may want to combine some fixed 
spatial pattern (such as an EOF) with a time-varying time series (the principal
component). Or you may want to combine results from a sensitivity analysis
where several parameters have been varied (one dimension per parameter). 
Here a minimal example where the above-define annual variable is combined with 
seasonally-varying data (camping summer and winter prices). 

Arrays are said to be **broadcast**: 

>>> seasonal_data = DimArray([10, 100], axes=[['winter','summer']], dims=['season'])
>>> combined_data = yearly_data * seasonal_data
>>> combined_data 
dimarray: 6 non-null elements (0 null)
dimensions: 'year', 'season'
0 / year (3): 1950 to 1970
1 / season (2): winter to summer
array([[  0,   0],
       [ 10, 100],
       [ 20, 200]])

..  _Dataset:

Dataset
-------

As a commodity, the **`Dataset`** class is an ordered dictionary of DimArrays which also maintains axis aligment

>>> dataset = Dataset({'combined_data':combined_data, 'yearly_data':yearly_data,'seasonal_data':seasonal_data})
>>> dataset
Dataset of 3 variables
dimensions: 'season', 'year'
0 / season (2): winter to summer
1 / year (3): 1950 to 1970
seasonal_data: ('season',)
combined_data: ('year', 'season')
yearly_data: ('year',)

It is one step away from creating a new DimArray from these various arrays, by broadcasting dimensions as needed:

>>> dataset.to_array(axis='variable')
dimarray: 18 non-null elements (0 null)
dimensions: 'variable', 'season', 'year'
0 / variable (3): seasonal_data to yearly_data
1 / season (2): winter to summer
2 / year (3): 1950 to 1970
array([[[ 10,  10,  10],
        [100, 100, 100]],
<BLANKLINE>
       [[  0,  10,  20],
        [  0, 100, 200]],
<BLANKLINE>
       [[  0,   1,   2],
        [  0,   1,   2]]])

Note that they are various ways of combining DimArray instances. In many case (when no dimension broadcasting is involved), it is simpler to just use the :py:func:`dimarray.stack` method.

..  _NetCDF_reading_and_writing:

NetCDF reading and writing
--------------------------

A natural I/O format for such an array is netCDF, common in geophysics, which rely on
the netCDF4 package. If netCDF4 is installed (much recommanded), a dataset can easily read and write to the netCDF format:

>>> dataset.write_nc('test.nc', mode='w')


>>> import dimarray as da
>>> da.read_nc('test.nc', 'combined_data')
dimarray: 6 non-null elements (0 null)
dimensions: 'year', 'season'
0 / year (3): 1950 to 1970
1 / season (2): winter to summer
array([[  0,   0],
       [ 10, 100],
       [ 20, 200]])

..  _Reshaping_arrays:

Reshaping arrays
----------------

Additional novelty includes methods to reshaping an array in easy ways, very useful for high-dimensional data analysis.

>>> large_array = da.array(np.arange(2*2*5*2).reshape(2,2,5,2), dims=('A','B','C','D'))
>>> small_array = large_array.group('A','B').group('C','D')  # same as reshape('A,B','C,D')
>>> small_array
dimarray: 40 non-null elements (0 null)
dimensions: 'A,B', 'C,D'
0 / A,B (4): (0, 0) to (1, 1)
1 / C,D (10): (0, 0) to (4, 1)
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]])

..  _interfacing_with_pandas:

interfacing with pandas
-----------------------

For things that pandas does better, such as pretty printing, I/O to many formats, and low-dimensional data analysis, just use the :py:meth:`dimarray.DimArray.to_pandas` method

>>> small_array.to_pandas()
C     0       1       2       3       4    
D     0   1   0   1   0   1   0   1   0   1
A B                                        
0 0   0   1   2   3   4   5   6   7   8   9
  1  10  11  12  13  14  15  16  17  18  19
1 0  20  21  22  23  24  25  26  27  28  29
  1  30  31  32  33  34  35  36  37  38  39

And :py:meth:`dimarray.DimArray.from_pandas` works to convert pandas objects to `DimArray` (also supports `MultiIndex`):

>>> import pandas as pd
>>> s = pd.DataFrame([[1,2],[3,4]], index=['a','b'], columns=[1950, 1960])
>>> da.from_pandas(s)
dimarray: 4 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (2): a to b
1 / x1 (2): 1950 to 1960
array([[1, 2],
       [3, 4]])

For more information, you can use inline help (help() or ?) or refer to :ref:`page_reference`