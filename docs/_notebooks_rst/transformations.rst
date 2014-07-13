.. This file was generated automatically from the ipython notebook:
.. notebooks/transformations.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_transformations:


Along-axis transformations
==========================
:download:`Download notebook </notebooks/transformations.ipynb>` 


.. _Basics:

Basics
------

Most numpy transformations are built in. Let's create some data to try it out:

>>> from dimarray import DimArray
>>> a = DimArray([[1,2,3],[4,5,6]], axes=[['a','b'], [2000,2001,2002]], dims=['time', 'items'])
>>> a
dimarray: 6 non-null elements (0 null)
0 / time (2): a to b
1 / items (3): 2000 to 2002
array([[1, 2, 3],
       [4, 5, 6]])

The classical numpy syntax names are used (`sum`, `mean`, `max`...) are used. For transformation that reduce an axis, the default behaviour is to flatten the array prior to the transformation, consistently with numpy:

>>> a.mean() # sum over all axes
3.5

But the `axis=` parameter can also be passed explicitly to reduce only a specific axis:

>>> a.mean(axis=0) # sum over first axis 
dimarray: 3 non-null elements (0 null)
0 / items (3): 2000 to 2002
array([ 2.5,  3.5,  4.5])

but it is now also possible to indicate axis name:

>>> a.mean(axis='time') # named axis
dimarray: 3 non-null elements (0 null)
0 / items (3): 2000 to 2002
array([ 2.5,  3.5,  4.5])

In addition, one can now provide a tuple to the `axis=` parameter, to reduce several axes at once

>>> a.mean(axis=('time','items')) #  named axis
3.5

Of course, the above example makes more sense when they are more than two axes. To perform an operation on the flatten array, the convention is to provide the `None` value for `axis=`, which is the default behaviour in all reduction operators.

..  note :: transformations that accumulate along an axis (`cumsum`, `cumprod`) default on the last axis (axis=-1) instead of flattening the array. This is also true of the `diff` operator. 

While most methods directly call numpy's in most cases, some subtle differences may exist that have to do with the need to define an axis values consistent with the operation. For example the :meth:`diff <dimarray.DimArray.diff>` method proposes several values for a `scheme` parameter ("centered", "backward", "forward"). Of interest also, the :meth:`argmin <dimarray.DimArray.argmin>` and :meth:`argmax <dimarray.DimArray.argmax>` methods return the value of the axis at the extrema instead of the integer position:

>>> a.argmin() 
('a', 2000)

...which is consistent with dimarray indexing:

>>> a[a.argmin()]
1

.. _Missing_values:

Missing values
--------------

`dimarray` treats `NaN` as missing values, which can be skipped in transformations by passing skipna=True. In the example below we use a float-typed array because there is no `NaN` type in integer arrays.

>>> import numpy as np
>>> a = DimArray([[1,2,3],[4,5,6]], dtype=float)
>>> a[1,2] = np.nan
>>> a
dimarray: 5 non-null elements (1 null)
0 / x0 (2): 0 to 1
1 / x1 (3): 0 to 2
array([[  1.,   2.,   3.],
       [  4.,   5.,  nan]])

>>> a.sum(axis=0)  # here the nans are not skipped
dimarray: 2 non-null elements (1 null)
0 / x1 (3): 0 to 2
array([  5.,   7.,  nan])

>>> a.sum(axis=0, skipna=True)
dimarray: 3 non-null elements (0 null)
0 / x1 (3): 0 to 2
array([ 5.,  7.,  3.])

.. _Weighted_mean,_std_and_var:

Weighted mean, std and var
--------------------------

These three functions check for the `weights` attribute of the axes they operate on. If different from `None` (the default), then the average is weighted according to `weights`. Here a practical example:

>>> np.random.seed(0) # to make results reproducible
>>> v = DimArray(np.random.rand(3,2), axes=[[-80, 0, 80], [-180, 180]], dims=['lat','lon'])


Classical, unweighted mean:

>>> v.mean()  
0.58019972362897432

Now we build a weight array as the cosine of the latitude (because of the sphericity of the Earth) and the smaller area of latitude bands at high latitudes:

>>> w = np.cos(np.radians(v.lat)) # weight based on latitude values 


We can pass this array of weights via the `weights` parameter:

>>> v.mean(weights={'lat':w})  # lat axis receives weights
0.57628879031663871

Or by setting the `weights` attribute to the "lat" axis, to make this change permanent:

>>> v.axes['lat'].weights = w


>>> v.mean()
0.57628879031663871

Weights are conserved by slicing and array-indexing:

>>> v[[0,80]].axes['lat'].weights
array([ 1.        ,  0.17364818])

Weights can also be defined as a function of axis values:

>>> v.axes['lat'].weights = lambda x : np.cos(np.radians(x))


>>> v.mean()
0.57628879031663871

Under the hood, weights are computed via the :meth:`DimArray._get_weights` method, so you can always check which weights are being used:

>>> v._get_weights()
dimarray: 6 non-null elements (0 null)
0 / lat (3): -80 to 80
1 / lon (2): -180 to 180
array([[ 0.17364818,  0.17364818],
       [ 1.        ,  1.        ],
       [ 0.17364818,  0.17364818]])

It is normally `None`:

>>> v.axes['lat'].weights = None
>>> v._get_weights()


The `GeoArray` class defines weights automatically for latitude:

>>> from dimarray.geo import GeoArray
>>> g = GeoArray(v, copy=True)
>>> g._get_weights()
dimarray: 6 non-null elements (0 null)
0 / lat (3): -80 to 80
1 / lon (2): -180 to 180
array([[ 0.17364818,  0.17364818],
       [ 1.        ,  1.        ],
       [ 0.17364818,  0.17364818]])