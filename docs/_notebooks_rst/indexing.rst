.. This file was generated automatically from the ipython notebook:
.. notebooks/indexing.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_indexing:


Advanced Indexing
=================
:download:`Download notebook </notebooks/indexing.ipynb>` 


Let's first define an array to test indexing

>>> from dimarray import DimArray


>>> v = DimArray([[1,2],[3,4],[5,6],[7,8]], axes=[["a","b","c","d"], [10.,20.]], dims=['x0','x1'], dtype=float) 
>>> v
dimarray: 8 non-null elements (0 null)
0 / x0 (4): 'a' to 'd'
1 / x1 (2): 10.0 to 20.0
array([[1., 2.],
       [3., 4.],
       [5., 6.],
       [7., 8.]])

.. _Basics__integer,_array,_slice:

Basics: integer, array, slice
-----------------------------

There are various ways of indexing a DimArray, and all follow numpy's rules, except that in the default behaviour indices refer to axis values and not to position on the axis, in contrast to numpy. 

>>> v['a',20]  # extract a single item
2.0

The `ix` attrubutes is the pendant for position (integer) indexing (and exclusively so !). It is therefore similar to indexing on the `values` attribute, except that it returns a new DimArray, where v.values[...] would return a numpy ndarray.

>>> v.ix[0,:]
dimarray: 2 non-null elements (0 null)
0 / x1 (2): 10.0 to 20.0
array([1., 2.])

Note that the last element of slices is INCLUDED, contrary to numpy's position indexing. Step argument is always intrepreted as an integer.

>>> v['a':'c',10]  # 'c' is INCLUDED
dimarray: 3 non-null elements (0 null)
0 / x0 (3): 'a' to 'c'
array([1., 3., 5.])

>>> v[['a','c'],10]  # it is possible to provide a list
dimarray: 2 non-null elements (0 null)
0 / x0 (2): 'a' to 'c'
array([1., 5.])

>>> v[v.x0 != 'b',10]  # boolean indexing is also fine
dimarray: 3 non-null elements (0 null)
0 / x0 (3): 'a' to 'd'
array([1., 5., 7.])

If several array-like indices are provided, "orthogonal" indexing is performed, along each dimension independently:

>>> v[['a','c'],[10,20]]  # it is possible to provide a list
dimarray: 4 non-null elements (0 null)
0 / x0 (2): 'a' to 'c'
1 / x1 (2): 10.0 to 20.0
array([[1., 2.],
       [5., 6.]])

 See below for the cases where you do need numpy-like index broadcasting, using the `take` method.

.. _Modify_array_values:

Modify array values
-------------------

All the above can be used to change array values, consistently with what you would expect. 

>>> v['a':'c',10] = 11
>>> v.ix[2, -1] = 22   # same as v.values[2, -1] = 44
>>> v[v == 2] = 33
>>> v[v.x0 == 'b', v.x1 == 20] = 44
>>> v
dimarray: 8 non-null elements (0 null)
0 / x0 (4): 'a' to 'd'
1 / x1 (2): 10.0 to 20.0
array([[11., 33.],
       [11., 44.],
       [11., 22.],
       [ 7.,  8.]])

.. _take_and_put_methods:

take and put methods
--------------------

These two methods :py:meth:`dimarray.DimArray.put` and :py:meth:`dimarray.DimArray.take` are the machinery to accessing and modifying items in the examples above.
They may be useful to use directly for generic programming. 
They are similar to numpy methods of the same name, but also work in multiple dimensions.
In particular, they both take dictionary, tuples and boolean arrays as `indices` argument.

>>> v = DimArray([[1,2],[3,4],[5,6],[7,8]], labels=[["a","b","c","d"], [10.,20.]], dims=['x0','x1'], dtype=float) 


>>> import numpy as np
>>> v[:,10]  # doctest: +SKIP
>>> v.take(10, axis=1)  # doctest: +SKIP
>>> v.take(10, axis='x1')  # doctest: +SKIP
>>> v.take({'x1':10}) # dict  # doctest: +SKIP
>>> v.take((slice(None),10)) # tuple # doctest: +SKIP
dimarray: 4 non-null elements (0 null)
0 / x0 (4): 'a' to 'd'
array([1., 3., 5., 7.])

The two latter forms, `tuple` or `dict`, allow performing multi-indexing. Array broadcasting is controlled by "broadcast" parameter.

>>> v.take({'x0':['a','b'], 'x1':[10, 20]}, broadcast=True) 
dimarray: 2 non-null elements (0 null)
0 / x0,x1 (2): ('a', '10.0') to ('b', '20.0')
array([1., 4.])

>>> v.take({'x0':['a','b'], 'x1':[10, 20]}, broadcast=False)  #  same as v.box[['a','b'],[10, 20]]
dimarray: 4 non-null elements (0 null)
0 / x0 (2): 'a' to 'b'
1 / x1 (2): 10.0 to 20.0
array([[1., 2.],
       [3., 4.]])

The 'indexing' parameter can be set to `position` (same as `ix`) instead of `values`

>>> v.take(0, axis=1, indexing='position')
dimarray: 4 non-null elements (0 null)
0 / x0 (4): 'a' to 'd'
array([1., 3., 5., 7.])

Note the `put` command modifies values in-place by default, unless `inplace=False`.

>>> v.put(indices=10, values=-99, axis='x1', inplace=False)
dimarray: 8 non-null elements (0 null)
0 / x0 (4): 'a' to 'd'
1 / x1 (2): 10.0 to 20.0
array([[-99.,   2.],
       [-99.,   4.],
       [-99.,   6.],
       [-99.,   8.]])