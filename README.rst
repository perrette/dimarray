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
the netCDF4 package. Other formats are under development (HDF5). Metadata are also 
supported, and conserved via slicing and along-axis transformations.

Notebook tutorial:
------------------
http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb

Download latest version on GitHub:
----------------------------------
https://github.com/perrette/dimarray/

Requires:
---------
- numpy

- netCDF4 (optional) :  for netCDF I/O, found at https://code.google.com/p/netcdf4-python/)

- matplotlib (optional) : for plotting (for now plot command also requires pandas)

- pandas (optional) :  to_pandas() and from_pandas() methods, plot()

Installation:
-------------
sudo python setup.py install

Get started
-----------


A **``DimArray``** can be defined just like a numpy array, with
additional information about its axes, which can be given via ``labels``
and ``dims`` parameters (see below another example with a list of tuples [(name_0, values_0), ...]).

>>> from dimarray import DimArray, Dataset
>>> a = DimArray(values=[[1,2,3],[4,5,6.]], labels=[["a","b"], [0,1,2]], dims=['dim0','dim1']) 
>>> a
dimarray: 6 non-null elements (0 null)
dimensions: 'dim0', 'dim1'
0 / dim0 (2): a to b
1 / dim1 (3): 0 to 2
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])


Array data are stored in a ``values`` **attribute**:

>>> a.values
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])


and axis names and values can be accessed straightforwardly, just like
``shape`` and ``ndim``:

>>> a.dims 
('dim0', 'dim1')
>>> a.labels   # same as (a.dim0, a.dim1)
(array(['a', 'b'], dtype=object), array([0, 1, 2]))


**Indexing** works on labels just as expected, including ``slice`` and
boolean array.

>>> a['b', 1]
5.0

but integer-index is always possible via ``ix`` toogle between
``labels``- and ``position``-based indexing:

>>> a.ix[1, 1]
5.0

Standard numpy **transformations** are defined, and now accept axis name:

>>> a.mean(axis='dim0')
dimarray: 3 non-null elements (0 null)
dimensions: 'dim1'
0 / dim1 (3): 0 to 2
array([ 2.5,  3.5,  4.5])


and can ignore **missing values (nans)** if asked to:

>>> import numpy as np
>>> a['a',2] = np.nan
>>> a
dimarray: 5 non-null elements (1 null)
dimensions: 'dim0', 'dim1'
0 / dim0 (2): a to b
1 / dim1 (3): 0 to 2
array([[  1.,   2.,  nan],
       [  4.,   5.,   6.]])

>>> a.mean(axis='dim0', skipna=True)
dimarray: 3 non-null elements (0 null)
dimensions: 'dim1'
0 / dim1 (3): 0 to 2
array([ 2.5,  3.5,  6. ])

More methods to handle missing values are available, though still experimental
at this stage, inspired by pandas: **dropna**, **setna**, **fillna**

Having axis name and axis values allow on-the-fly **axis alignment** and
**dimension broadcasting** in basic operations (addition, etc...), so
that rules can be defined for nearly every sequence of operands.

Let's define some axes on dimensions ``time`` and ``items``, using the
list-of-tuple form [(name, values), (name2, values2), ...], here reduced to simple `tuple` since
we are dealing with 1-D arrays.

>>> time = ('time', [1950, 1951, 1952])
>>> incomplete_time = ('time', [1950, 1952])
>>> items = ('items', ['a','b'])

see how two arrays with different time indices align, and how the
missing year in the second array is replaced by nan:

>>> timeseries = DimArray([1,2,3], time)
>>> incomplete_timeseries = DimArray([4, 5], incomplete_time)
>>> timeseries + incomplete_timeseries
dimarray: 2 non-null elements (1 null)
dimensions: 'time'
0 / time (3): 1950 to 1952
array([  5.,  nan,   8.])


If one of the operands lacks a dimension, it is automatically repeated
(broadcast) to match the other operand's shape. In this example, an
array of weights is fixed in time, whereas the data to be weighted
changes at each time step.

>>> data = DimArray([[1,2,3],[40,50,60]], [items, time])
>>> weights = DimArray([2, 0.5], items)
>>> data * weights
dimarray: 6 non-null elements (0 null)
dimensions: 'items', 'time'
0 / items (2): a to b
1 / time (3): 1950 to 1952
array([[  2.,   4.,   6.],
       [ 20.,  25.,  30.]])


As a commodity, the **``Dataset``** class is an ordered dictionary of
DimArrays which also maintains axis aligment

>>> dataset = Dataset(data=data, weights=weights, incomplete_timeseries=incomplete_timeseries)
>>> dataset
Dataset of 3 variables
dimensions: 'items', 'time'
0 / items (2): a to b
1 / time (3): 1950 to 1952
weights: ('items',)
incomplete_timeseries: ('time',)
data: ('items', 'time')



It is one step away from creating a new DimArray from these various
arrays, by broadcasting dimensions as needed:

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



Note a shorter way of obtaining the above, if the only desired result is
to align axes, would have been to use the **``from_arrays``** method (see interactive help).

A natural I/O format for such an array is netCDF, common in geophysics,
which rely on the netCDF4 package. If netCDF4 is installed (much
recommanded), a dataset can easily read and write to the netCDF format:

>>> dataset.write_nc('test.nc', mode='w')
>>> import dimarray as da
>>> da.read_nc('test.nc', 'incomplete_timeseries')
dimarray: 2 non-null elements (1 null)
dimensions: 'time'
0 / time (3): 1950 to 1952
array([  4.,  nan,   5.])


Additional novelty includes a wealth of methods to reshape an array via 
dimension names, such as **group**, **ungroup**, **reshape**, **transpose**, **swapaxes**
all very useful for high-dimensional data analysis.

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


And for things that pandas does better, such as pretty printing, I/O to
many formats, and low-dimensional data analysis, just use the
**``to_pandas``** method (see reverse **``from_pandas``**):

>>> small_array.to_pandas()
C     0       1       2       3       4    
D     0   1   0   1   0   1   0   1   0   1
A B                                        
0 0   0   1   2   3   4   5   6   7   8   9
  1  10  11  12  13  14  15  16  17  18  19
1 0  20  21  22  23  24  25  26  27  28  29
  1  30  31  32  33  34  35  36  37  38  39


More on the notebook documentation:
http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb


Comparisons with other packages:
--------------------------------

- **pandas**
            ...is an excellent package for low-dimensional data analysis, 
            with many I/O features, but is mostly limited to 2 dimensions (DataFrame), 
            or up to 4 dimensions (Panel, Panel4D). `dimarray` includes
            some of the nice `pandas` features, such as indexing on axis values, 
            automatic axis alignment, intuitive string representation,
            or a parameter to ignore nans in axis reduction operations. 
            `dimarray` extends these functionalities to any number 
            of dimensions. In general, `dimarray` is designed to be more consistent with 
            `numpy`'s ndarray, whereas `pandas` is somewhat between a dictionary and 
            a numpy array. One consequence is that standard indexing with `[]` can be 
            multi-dimensional, another is that iteration is on sub-arrays and not on 
            axis values (the keys). `dimarray` comes with `to_pandas` and `from_pandas`
            methods to use the most of each of the packages (also supports `MultiIndex`
            via the equivalent `GroupedAxis` object). For convenience, a `plot`
            method is defined in `dimarray` as an alias for to_pandas().plot().

- **larry** 
    ...was pioneer as labelled array, it skips nans in along-axis transforms
    and comes with a wealth of built-in methods. It seems to do a very good job but does
    not offer as many features as `dimarray` or `pandas` as far as indexing 
    is concerned. For example it does not 
    support naming dimensions and does not support indexing on axis values 
    (but provides `get` method for single element access). Still, from the 
    structure (array-like), `dimarray` is closer to larry than to pandas.
    

Compared with these two pacakges, `dimarray` adds the possibility of passing axis 
name to the various methods, instead of simply axis rank. Having a focus on dimension
names and axis values instead of axis rank and position of elements along an axis
is a strong feature of `dimarray`. This applies for
instance to along-axis operation, `take` and `put` methods, or reshaping operations.
Additionally, `dimarray` is to my knowledge the only package supporting automatic
dimension broadcasting for any two operands. This has proven useful to write pretty
generic code with arrays of various shape which all share a few dimensions `time`, 
`lon`, `lat`, `model`, `scenario`, `sample`, `percentile` and so on.

- **iris** 
    ...looks like a very powerful package to manipulate geospatial data with 
    metadata, netCDF I/O, performing grid transforms etc..., but it is quite a jump 
    from numpy's `ndarray` and requires a bit of learning. 
    In contrast, `dimarray` is more general and intuitive for python users. `dimarray`
    also comes with netCDF I/O capability and may gain a few geospatial features 
    (weighted mean for lon/lat, 360 modulo for lon, regridding, etc...) as a subpackage 
    **dimarray.geo** -- and why not an interface to `iris`.


Further development:
--------------------
All suggestions for improvement very welcome, please file an `issue` on github:
https://github.com/perrette/dimarray/ for further discussion.
