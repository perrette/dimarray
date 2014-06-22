.. This file was generated automatically from the ipython notebook:
.. notebooks/transformations.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

..  _page_transformations:


Along-axis transformations
==========================

..  _Use_axis_name:

Use axis name
-------------

Most numpy transformations are built in. Let's create some data to try it out:

>>> import dimarray as da
>>> import numpy as np
>>> values = np.arange(4*3).reshape(4,3)
>>> time = 'time', np.arange(1950,1954) 
>>> lat = 'lat', np.linspace(-90,90,3)
>>> v = da.DimArray(values, axes=[np.arange(1950,1954), np.linspace(-90,90,3)], dims=['time','lat'])
>>> v
dimarray: 12 non-null elements (0 null)
dimensions: 'time', 'lat'
0 / time (4): 1950 to 1953
1 / lat (3): -90.0 to 90.0
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])

In axis-reduction operations, not providing the axis results in the operation being applied to the flattened array, following numpy's rule:

>>> v.sum() # sum over all axes
66

To perform the operation along an axis, it must be indicated by an integer (axis rank) or - new in dimarray - by a string (axis name):

>>> s = v.sum(axis=0) # sum over first axis
>>> s = v.sum(axis='time') # named axis
>>> s
dimarray: 3 non-null elements (0 null)
dimensions: 'lat'
0 / lat (3): -90.0 to 90.0
array([18, 22, 26])

All axis-reduction transformations, following numpy:
sum, prod, mean, var, std, median, min, max, php, argmin, argmax, all, any

In `dimarray`, the `argmin` and `argmax` functions return axis value instead of axis position.

>>> v.argmin() 
(1950, -90.0)

...which is consistent with indexing on axis values:

>>> v[v.argmin()], v.min() 
(0, 0)

The along axis version works similarly:

>>> date_min = v.argmin(axis='time') 
>>> date_min
dimarray: 3 non-null elements (0 null)
dimensions: 'lat'
0 / lat (3): -90.0 to 90.0
array([1950, 1950, 1950])

>>> v[date_min, v.lat]  # this makes use of array-broadcasting when indexing with two arrays
dimarray: 3 non-null elements (0 null)
dimensions: 'time,lat'
0 / time,lat (3): (1950.0, -90.0) to (1950.0, 90.0)
array([0, 1, 2])

Operations that accumulate along an axis are also implemented, by default along the last axis, consistently with numpy. Let's use a simpler 1-D example here.

>>> v = da.DimArray(np.arange(1,5), time, dtype=float)
>>> v
dimarray: 4 non-null elements (0 null)
dimensions: 'time'
0 / time (4): 1950 to 1953
array([ 1.,  2.,  3.,  4.])

>>> p = v.cumprod()
>>> s = v.cumsum()
>>> s
dimarray: 4 non-null elements (0 null)
dimensions: 'time'
0 / time (4): 1950 to 1953
array([  1.,   3.,   6.,  10.])

A new `diff` method comes with `dimarray`, which reduces axis size by one, by default (and by default `diff` operates along the last axis, like `cumsum`).

>>> s.diff()
dimarray: 3 non-null elements (0 null)
dimensions: 'time'
0 / time (3): 1951 to 1953
array([ 2.,  3.,  4.])

The `keepaxis=` parameter fills array with `nan` where necessary to keep the axis unchanged. Default is backward differencing: `diff[i] = v[i] - v[i-1]`.

>>> s.diff(keepaxis=True)
dimarray: 3 non-null elements (1 null)
dimensions: 'time'
0 / time (4): 1950 to 1953
array([ nan,   2.,   3.,   4.])

But other schemes are available to control how the new axis is defined: `backward` (default), `forward` and even `centered`

>>> s.diff(keepaxis=True, scheme="forward") # diff[i] = v[i+1] - v[i]
dimarray: 3 non-null elements (1 null)
dimensions: 'time'
0 / time (4): 1950 to 1953
array([  2.,   3.,   4.,  nan])

The `keepaxis=True` option is invalid with the `centered` scheme, since every axis value is modified by definition:

>>> s.diff(axis='time', scheme='centered')
dimarray: 3 non-null elements (0 null)
dimensions: 'time'
0 / time (3): 1950.5 to 1952.5
array([ 2.,  3.,  4.])

..  _Missing_values:

Missing values
--------------

`dimarray` treats `nan` as missing values, which can be skipped in transformations by passing skipna=True. Note that `nan` is has a `float` type so it cannot be assigned to an integer array.

>>> import numpy as np
>>> import dimarray as da


>>> a = da.DimArray([[1,2,3],[4,5,6]], dtype=float)
>>> a[1,2] = np.nan
>>> a
dimarray: 5 non-null elements (1 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (3): 0 to 2
array([[  1.,   2.,   3.],
       [  4.,   5.,  nan]])

>>> a.mean(axis=0)
dimarray: 2 non-null elements (1 null)
dimensions: 'x1'
0 / x1 (3): 0 to 2
array([ 2.5,  3.5,  nan])

>>> a.mean(axis=0, skipna=True)
dimarray: 3 non-null elements (0 null)
dimensions: 'x1'
0 / x1 (3): 0 to 2
array([ 2.5,  3.5,  3. ])

A few other methods exist but are experimental. They are mere aliases for classical `a[np.isnan[a]] = value` syntax, but automatically coerce integer type to float, and perform a copy by default. This could also be useful in the future to define a missing value flag other than `nan`, for example when working with integer array.

>>> a.fillna(99)
dimarray: 6 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (3): 0 to 2
array([[  1.,   2.,   3.],
       [  4.,   5.,  99.]])

`setna` can also be provided with a list of values (or boolean arrays) to set to nan:

>>> b = a.setna([1,4])
>>> b
dimarray: 3 non-null elements (3 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (3): 0 to 2
array([[ nan,   2.,   3.],
       [ nan,   5.,  nan]])

More interestingly, the `dropna` methods helps getting rid of nans arising in grouping operations, similarly to `pandas`:

>>> b.dropna(axis=1)
dimarray: 2 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (1): 1 to 1
array([[ 2.],
       [ 5.]])

But in some cases, you are still ok with a certain number of nans, but want to have a minimum of 1 or more valid values:

>>> b.dropna(axis=1, minvalid=1)  # minimum number of valid values, equivalent to `how="all"` in pandas
dimarray: 3 non-null elements (1 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (2): 1 to 2
array([[  2.,   3.],
       [  5.,  nan]])

..  _Weighted_mean_[experimental]:

Weighted mean [experimental]
----------------------------

Each axis can have a `weights` attribute. If not None, it will be automatically used when computing mean, var, std

>>> a = da.DimArray(np.arange(5))
>>> a
dimarray: 5 non-null elements (0 null)
dimensions: 'x0'
0 / x0 (5): 0 to 4
array([0, 1, 2, 3, 4])

The mean as you'd expect:

>>> a.mean()  #  standard mean
2.0

Now adding the weight parameter:

>>> a.axes[0].weights = [0, 0, 0, 1, 0]
>>> a.mean()
3.0

Note it is preserved via indexing 

>>> a.ix[-3:].axes[0].weights
array([0, 1, 0])

Also possible as a function:

>>> a.axes[0].weights = lambda x: x**2
>>> a.mean()
3.3333333333333335