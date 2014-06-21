.. This file was generated automatically from the ipython notebook:
.. notebooks/getting_started.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

..  _getting_started:


Getting started
===============

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

For more information refer to section on :ref:`data_structure`

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

..  _data_alignment__automatic_broadcasting_and_reindexing:

data alignment: automatic broadcasting and reindexing
-----------------------------------------------------

Having axis name and axis values allow on-the-fly **axis alignment** and
**dimension broadcasting** in basic operations (addition, etc...),
so that rules can be defined for nearly every sequence of operands.

Let's define some axes on dimensions `time` and `items`, using the tuple form (name, values)

>>> time = ('time', [1950, 1951, 1952])
>>> incomplete_time = ('time', [1950, 1952])
>>> items = ('items', ['a','b'])


see how two arrays with different time indices align, and how the missing year in the second array is replaced by nan:

>>> timeseries = DimArray([1,2,3], time)
>>> incomplete_timeseries = DimArray([4, 5], incomplete_time)
>>> timeseries + incomplete_timeseries
dimarray: 2 non-null elements (1 null)
dimensions: 'time'
0 / time (3): 1950 to 1952
array([  5.,  nan,   8.])

If one of the operands lacks a dimension, it is automatically repeated (broadcast) to match the other operand's shape. In this example, an array of weights is fixed in time, whereas the data to be weighted changes at each time step. 

>>> data = DimArray([[1,2,3],[40,50,60]], [items, time])
>>> weights = DimArray([2, 0.5], items)
>>> 
>>> data * weights
dimarray: 6 non-null elements (0 null)
dimensions: 'items', 'time'
0 / items (2): a to b
1 / time (3): 1950 to 1952
array([[  2.,   4.,   6.],
       [ 20.,  25.,  30.]])

..  _Dataset:

Dataset
-------

As a commodity, the **`Dataset`** class is an ordered dictionary of DimArrays which also maintains axis aligment

>>> dataset = Dataset({'data':data, 'weights':weights,'incomplete_timeseries':incomplete_timeseries})
>>> dataset
Dataset of 3 variables
dimensions: 'items', 'time'
0 / items (2): a to b
1 / time (3): 1950 to 1952
weights: ('items',)
incomplete_timeseries: ('time',)
data: ('items', 'time')

It is one step away from creating a new DimArray from these various arrays, by broadcasting dimensions as needed:

>>> dataset.to_array(axis='variables')
dimarray: 16 non-null elements (2 null)
dimensions: 'variables', 'items', 'time'
0 / variables (3): weights to data
1 / items (2): a to b
2 / time (3): 1950 to 1952
array([[[  2. ,   2. ,   2. ],
        [  0.5,   0.5,   0.5]],
<BLANKLINE>
       [[  4. ,   nan,   5. ],
        [  4. ,   nan,   5. ]],
<BLANKLINE>
       [[  1. ,   2. ,   3. ],
        [ 40. ,  50. ,  60. ]]])

Note a shorter way of obtaining the above, if the only desired result is to align axes, would have been to use the **`stack`** method (see interactive help).

..  _NetCDF_reading_and_writing:

NetCDF reading and writing
--------------------------

A natural I/O format for such an array is netCDF, common in geophysics, which rely on
the netCDF4 package. If netCDF4 is installed (much recommanded), a dataset can easily read and write to the netCDF format:

>>> dataset.write_nc('test.nc', mode='w')


>>> import dimarray as da
>>> da.read_nc('test.nc', 'incomplete_timeseries')
dimarray: 2 non-null elements (1 null)
dimensions: 'time'
0 / time (3): 1950 to 1952
array([  4.,  nan,   5.])

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

And for things that pandas does better, such as pretty printing, I/O to many formats, and low-dimensional data analysis, just use the **`to_pandas`** method (see reverse **`from_pandas`**):

>>> print small_array.to_pandas()
C     0       1       2       3       4    
D     0   1   0   1   0   1   0   1   0   1
A B                                        
0 0   0   1   2   3   4   5   6   7   8   9
  1  10  11  12  13  14  15  16  17  18  19
1 0  20  21  22  23  24  25  26  27  28  29
  1  30  31  32  33  34  35  36  37  38  39
