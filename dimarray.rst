
Introduction to dimarray
========================


Download dimarray on `github <https://github.com/perrette/dimarray/>`_
or just take a look at this notebook on
`nbviewer <http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb>`_

Table of Content
~~~~~~~~~~~~~~~~

-  `Definition and attributes of a
   DimArray <#Definition-and-attributes-of-a-DimArray>`_
-  `Metadata <#Metadata>`_
-  `Indexing <#Indexing>`_

   -  `Basics: integer, array, slice <#Basics:-integer,-array,-slice>`_
   -  `Modify array values <#Modify-array-values>`_
   -  ```take`` and ``put`` methods <#take-and-put-methods>`_

-  `Reindex Axis <#Reindex-Axis>`_
-  `Missing values <#Missing-values>`_
-  `Modify dimensions: basic
   functionality <#Modify-dimensions:-basic-functionality>`_

   -  `Transpose <#Transpose>`_
   -  `Insert new axis <#Insert-new-axis>`_
   -  `Reshape <#Reshape>`_
   -  `Repeat <#Repeat>`_
   -  `Broadcast <#Broadcast>`_

-  `Align arrays <#Align-arrays>`_

   -  `Align dimensions <#Align-dimensions>`_
   -  `Align axes <#Align-axes>`_

-  `Join arrays <#Join-arrays>`_

   -  `Concatenate arrays along existing
      axis <#Concatenate-arrays-along-existing-axis>`_
   -  `From a collection of DimArray
      objects <#From-a-collection-of-DimArray-objects>`_
   -  `Aggregate arrays of varying
      dimensions <#Aggregate-arrays-of-varying-dimensions>`_

-  `Operations <#Operations>`_

   -  `Basic Operations <#Basic-Operations-------->`_
   -  `Operation with data alignment <#Operation-with-data-alignment->`_

-  `Numpy Transformations <#Numpy-Transformations>`_
-  `Dataset <#Dataset>`_
-  `NetCDF I/O <#NetCDF-I/O>`_

   -  `From a DimArray <#From-a-DimArray>`_
   -  `From/To Dataset <#From/To-Dataset>`_

-  `Experimental Features <#Experimental-Features>`_

   -  `Group / Ungroup <#Group-/-Ungroup>`_
   -  `Weighted mean <#Weighted-mean>`_
   -  `Interpolation <#Interpolation>`_
   -  `Export to other formats <#Export-to-other-formats>`_

-  `doctest framework <#doctest-framework>`_


Definition and attributes of a DimArray
---------------------------------------


A **``DimArray``** can be defined just like a numpy array, with
additional information about its dimensions, which are referred to as
``axes``, for consistency with numpy and pandas. The default way it to
provide axes as a list of tuples (``axis name``, ``axis values``) to
fully identify the array.

.. code:: python

    import numpy as np
    import dimarray as da
    from dimarray import DimArray
.. code:: python

    a = DimArray([[1,2,3],[4,5,6]], axes=[("dim0",["a","b"]), ("dim1", [0,1,2])]) 
    a



.. parsed-literal::

    dimarray: 6 non-null elements (0 null)
    dimensions: 'dim0', 'dim1'
    0 / dim0 (2): a to b
    1 / dim1 (3): 0 to 2
    array([[1, 2, 3],
           [4, 5, 6]])



Array data are stored in a ``values`` attribute:

.. code:: python

    a.values



.. parsed-literal::

    array([[1, 2, 3],
           [4, 5, 6]])



While dimensions are stored in an ``axes`` attribute, which is a custom
list of **``Axis``** objects:

.. code:: python

    a.axes



.. parsed-literal::

    dimensions: 'dim0', 'dim1'
    0 / dim0 (2): a to b
    1 / dim1 (3): 0 to 2



.. code:: python

    ax = a.axes[0]  # by integer position
    ax = a.axes['dim0'] # by axis name (for pythonistas: list with overloaded __getitem__ property)
    ax



.. parsed-literal::

    dim0 (2): a to b



An **``Axis``** object itself has ``name`` and ``values`` attributes:

.. code:: python

    ax.name



.. parsed-literal::

    'dim0'



.. code:: python

    ax.values



.. parsed-literal::

    array(['a', 'b'], dtype=object)



For convenience, axis names and values can be accessed directly via
``dims`` and ``labels`` attributes, and directly by their names (as long
as the name does not conflict with another protected attribute of the
class, in that case it needs to be accessed by axes[].values):

.. code:: python

    a.dims # alias for a.axes[0].name, a.axes[1].name



.. parsed-literal::

    ('dim0', 'dim1')



.. code:: python

    a.dim0, a.dim1   # alias for a.axes['dim0'].values, a.axes['dim1'].values
    a.labels



.. parsed-literal::

    (array(['a', 'b'], dtype=object), array([0, 1, 2]))



Note that numpy-like attribute ``shape`` and ``ndim``, among others, are
also defined:

.. code:: python

    a.shape



.. parsed-literal::

    (2, 3)



.. code:: python

    a.ndim



.. parsed-literal::

    2



In another world (like in R), one could also have chosen ``dims``
instead of ``axes``. ``numpy`` has decided otherwise and was followed by
``pandas``, so ``dimarray`` will just stick to it to reduce confusion.

The convention chosen in ``dimarray`` is to refer to axis names as
``dims`` (for dimensions) and to axis values as ``labels``. This choice
may seem a bit arbitrary, and to a certain extent it is, but it also has
some internal logics. In particular, ``dimensions`` in the physical
sense of the term refers to things with units, such time or space
dimensions. A dimension is more like a fundamental property of an axis.
It is conserved by indexing or slicing along an axis, and it determines
whether two axes can be aligned or concatenated. ``labels`` may be more
awkward when actually thinking about axis ``values``, but it makes full
sense when realizing that axis values serve as labelling the elements of
an array elements. There is even a handy package whose name is drawn
from it (``larry``, for labelled array).

Well, in the hope it makes some sense to you, let's go on.

For consistency with the wording ``dims``, ``labels`` to refer to axis
names and values, the alternative definition below is equivalent to the
one introduced at the beginning of this tutorial:

.. code:: python

    a = DimArray(values=[[1,2,3],[4,5,6]], labels=[["a","b"], [0,1,2]], dims=['dim0','dim1']) 
    a



.. parsed-literal::

    dimarray: 6 non-null elements (0 null)
    dimensions: 'dim0', 'dim1'
    0 / dim0 (2): a to b
    1 / dim1 (3): 0 to 2
    array([[1, 2, 3],
           [4, 5, 6]])



Note that if any of ``axes=``, ``dims=`` or ``labels=`` is omitted,
dimarray proceeds to automatic naming / labelling, using np.arange() for
axis values, and "x0", "x1" etc... for axis names:

.. code:: python

    a = DimArray(values=[[1,2,3],[4,5,6]], dims=['dim1','dim1']) # axis values defined as np.arange()
    a.labels



.. parsed-literal::

    (array([0, 1]), array([0, 1, 2]))



.. code:: python

    a = DimArray(values=[[1,2,3],[4,5,6]], labels=[['a','b'],[1,2,3]]) # axis values defined as np.arange()
    a.dims



.. parsed-literal::

    ('x0', 'x1')



As a convenience for the 1-D case when axis name is less relevant, the
brackets on ``labels`` can be omitted, with or without keywords:

.. code:: python

    a = DimArray(values=[1,6], labels=['a', 'b']) 
    a = DimArray([1,6], ['a', 'b']) 
    a



.. parsed-literal::

    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (2): a to b
    array([1, 6])



Metadata
--------


``DimArray`` and ``Axis`` objects, support metadata. They can be passed
by keyword arguments to DimArray (not via da.array\_kw or
DimArray.from\_kw !)

.. code:: python

    a = DimArray([[1,2,3],[4,5,6]])
    a.name='myname'
    a.units='myunits'
.. code:: python

    ax = a.axes[0]
    ax.units = "meters"
metadata are conserved by slicing and along-axis transformation, but are
lost with any other transformation

.. code:: python

    a[:].units



.. parsed-literal::

    'myunits'



.. code:: python

    ax[:].units



.. parsed-literal::

    'meters'



Indexing
--------


Basics: integer, array, slice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


There are various ways of indexing a DimArray, and all follow numpy's
rules, except that in the default behaviour indices refer to axis values
and not to position on the axis, in contrast to numpy.

.. code:: python

    from dimarray import DimArray
    import numpy as np
.. code:: python

    v = DimArray([[1,2],[3,4],[5,6],[7,8]], labels=[["a","b","c","d"], [10.,20.]], dims=['x0','x1'], dtype=float) 
    v



.. parsed-literal::

    dimarray: 8 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (4): a to d
    1 / x1 (2): 10.0 to 20.0
    array([[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.],
           [ 7.,  8.]])



.. code:: python

    v['a',20]  # extract a single item



.. parsed-literal::

    2.0



The ``ix`` attrubutes is the pendant for position (integer) indexing
(and exclusively so !). It is therefore similar to indexing on the
``values`` attribute, except that it returns a new DimArray, where
v.values[...] would return a numpy ndarray.

.. code:: python

    v.ix[0, 1] # or use `ix` to use integer indexing



.. parsed-literal::

    2.0



Note that the last element of slices is INCLUDED, contrary to numpy's
position indexing. Step argument is always intrepreted as an integer.

.. code:: python

    v['a':'c',10]  # 'c' is INCLUDED



.. parsed-literal::

    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): a to c
    array([ 1.,  3.,  5.])



.. code:: python

    v[['a','c'],10]  # it is possible to provide a list



.. parsed-literal::

    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (2): a to c
    array([ 1.,  5.])



.. code:: python

    v[v.x0 != 'b',10]  # boolean indexing is also fine



.. parsed-literal::

    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): a to d
    array([ 1.,  5.,  7.])



If several array-like indices are provided, they are broadcast into a
single shape (like numpy does), and values are extracted along the
corresponding line.

.. code:: python

    v[['a','c'],[10,20]]  # it is possible to provide a list



.. parsed-literal::

    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0,x1'
    0 / x0,x1 (2): ('a', '10.0') to ('c', '20.0')
    array([ 1.,  6.])



This is in contrast to matlab or pandas, which use box-like indexing,
along each dimension independently. This can be achieved with the
``box`` attribute:

.. code:: python

    v.box[['a','c'],[10,20]]  # indexing on each dimension, individually



.. parsed-literal::

    dimarray: 4 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): a to c
    1 / x1 (2): 10.0 to 20.0
    array([[ 1.,  2.],
           [ 5.,  6.]])



Modify array values
~~~~~~~~~~~~~~~~~~~


All the above can be used to change array values, consistently with what
you would expect. A few examples:

.. code:: python

    v[:] = 0
    v['d'] = 1
    v['b', 10] = 2
    v.box[['a','c'],[10,20]] = 3
    v[['a','c'],[10,20]] = 4
    v.values[-1] = 5 # last element to 5 
    v.ix[-1] = 6
    v



.. parsed-literal::

    dimarray: 8 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (4): a to d
    1 / x1 (2): 10.0 to 20.0
    array([[ 4.,  3.],
           [ 2.,  0.],
           [ 3.,  4.],
           [ 6.,  6.]])



``take`` and ``put`` methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


These two methods are the machinery to accessing and modifying items in
the examples above. They may be useful to use directly for generic
programming. They are similar to numpy methods of the same name, but
also work in multiple dimensions. In particular, they both take
dictionary, tuples and boolean arrays as ``indices`` argument.

.. code:: python

    v = DimArray([[1,2],[3,4],[5,6],[7,8]], labels=[["a","b","c","d"], [10.,20.]], dims=['x0','x1'], dtype=float) 
.. code:: python

    a = v[:,10]
    b = v.take(10, axis=1)
    c = v.take(10, axis='x1')
    d = v.take({'x1':10}) # dict
    e = v.take((slice(None),10)) # tuple
    assert(np.all(a==b) and np.all(a==b) and np.all(a==c) and np.all(a==d) and np.all(a==e))
    a



.. parsed-literal::

    dimarray: 4 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (4): a to d
    array([ 1.,  3.,  5.,  7.])



The two latter forms, ``tuple`` or ``dict``, allow performing
multi-indexing. Array broadcasting is controlled by "broadcast\_arrays"
parameter.

.. code:: python

    v.take({'x0':['a','b'], 'x1':[10, 20]}) 



.. parsed-literal::

    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0,x1'
    0 / x0,x1 (2): ('a', '10.0') to ('b', '20.0')
    array([ 1.,  4.])



.. code:: python

    v.take({'x0':['a','b'], 'x1':[10, 20]}, broadcast_arrays=False)  #  same as v.box[['a','b'],[10, 20]]



.. parsed-literal::

    dimarray: 4 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): a to b
    1 / x1 (2): 10.0 to 20.0
    array([[ 1.,  2.],
           [ 3.,  4.]])



The 'indexing' parameter can be set to ``position`` (same as ``ix``)
instead of ``values``

.. code:: python

    v.take(0, axis=1, indexing='position')



.. parsed-literal::

    dimarray: 4 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (4): a to d
    array([ 1.,  3.,  5.,  7.])



Note the ``put`` command returns a copy by default (``inplace=`` can be
passed as True, though).

.. code:: python

    v.put(-99, indices=10, axis='x1')



.. parsed-literal::

    dimarray: 8 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (4): a to d
    1 / x1 (2): 10.0 to 20.0
    array([[-99.,   2.],
           [-99.,   4.],
           [-99.,   6.],
           [-99.,   8.]])



Reindex Axis
------------


.. code:: python

    #import dim
    import dimarray.core._indexing as re; reload(re)
    import dimarray as da
    a = da.DimArray([1,2,3],[('x0', [1,2,3])])
    b = da.DimArray([3,4],[('x0',[1,3])])
    b.reindex_axis([1,2,3])



.. parsed-literal::

    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 1 to 3
    array([  3.,  nan,   4.])



.. code:: python

    # Can also reindex in "interp" mode
    b.reindex_axis([0,1,2,3], method='interp')



.. parsed-literal::

    dimarray: 3 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (4): 0 to 3
    array([ nan,  3. ,  3.5,  4. ])



.. code:: python

    # Or like another array
    c = da.DimArray([[1,2,3], [1,2,3]],[('x1',["a","b"]),('x0',[1, 2, 3])])
    b.reindex_like(c, method='interp')
    #b.reindex_axis([1,2,3], method='interp')



.. parsed-literal::

    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 1 to 3
    array([ 3. ,  3.5,  4. ])



Also works with string indices

.. code:: python

    a = da.DimArray([1,2,3],[('x0', ['a','b','c'])])
    a.reindex_axis(['b','d'])



.. parsed-literal::

    dimarray: 1 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (2): b to d
    array([  2.,  nan])



Missing values
--------------


.. code:: python

    import dimarray as da
    import numpy as np
    a = da.DimArray([[1,2,-99]])
    a.setna([-99,2])  
    #a.setna([-99, a>1])  # multi-dim, multi-values, boolean



.. parsed-literal::

    dimarray: 1 non-null elements (2 null)
    dimensions: 'x0', 'x1'
    0 / x0 (1): 0 to 0
    1 / x1 (3): 0 to 2
    array([[  1.,  nan,  nan]])



Modify dimensions: basic functionality
--------------------------------------


Basic numpy methods to modify array dimensions are implemented in
dimarray, with some additional functionality allowed by named
dimensions.

Transpose
~~~~~~~~~


Transpose, just like its numpy equivalent, permutes dimensions, but in
dimarray it can be provided with axis names instead of just axis
position.

.. code:: python

    a = DimArray([[1,2,3],[3,4,5]],dims=('x0','x1'))
    a.transpose()
    a.T



.. parsed-literal::

    dimarray: 6 non-null elements (0 null)
    dimensions: 'x1', 'x0'
    0 / x1 (3): 0 to 2
    1 / x0 (2): 0 to 1
    array([[1, 3],
           [2, 4],
           [3, 5]])



.. code:: python

    a = DimArray([[[1,2,3],[3,4,5]]],dims=('x2','x0','x1'))
    a.transpose('x1','x2','x0')



.. parsed-literal::

    dimarray: 6 non-null elements (0 null)
    dimensions: 'x1', 'x2', 'x0'
    0 / x1 (3): 0 to 2
    1 / x2 (1): 0 to 0
    2 / x0 (2): 0 to 1
    array([[[1, 3]],
    
           [[2, 4]],
    
           [[3, 5]]])



Insert new axis
~~~~~~~~~~~~~~~


Numpy provides a np.newaxis constant (equal to None), to augment the
array dimensions with new singleton axes. In dimarray, newaxis has been
implemented as an array method, which requires to indicate axis name and
optionally axis position (``pos=``). Under the ``repeat`` section,
you'll see it is also possible to input the values of the new axis in
order to repeast the array along it.

.. code:: python

    a = DimArray([1,2])
    a.newaxis('new', pos=1)  # singleton



.. parsed-literal::

    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0', 'new'
    0 / x0 (2): 0 to 1
    1 / new (1): None to None
    array([[1],
           [2]])



Reshape
~~~~~~~


.. code:: python

    a = da.DimArray.from_kw(np.arange(2), lon=[30., 40.])
    b = a.reshape(('time','lon'))
    b



.. parsed-literal::

    dimarray: 2 non-null elements (0 null)
    dimensions: 'time', 'lon'
    0 / time (1): None to None
    1 / lon (2): 30.0 to 40.0
    array([[0, 1]])



Repeat
~~~~~~


.. code:: python

    ### Single axis:
    >>> b.repeat(np.arange(1950,1955), axis="time")  # doctest: +ELLIPSIS



.. parsed-literal::

    dimarray: 10 non-null elements (0 null)
    dimensions: 'time', 'lon'
    0 / time (5): 1950 to 1954
    1 / lon (2): 30.0 to 40.0
    array([[0, 1],
           [0, 1],
           [0, 1],
           [0, 1],
           [0, 1]])



.. code:: python

    ### Multi-axis
    # ...create some dummy data:
    lon = np.linspace(10, 30, 2)
    lat = np.linspace(10, 50, 3)
    time = np.arange(1950,1955)
    ts = da.DimArray.from_kw(np.arange(5), time=time)
    cube = da.DimArray.from_kw(np.zeros((3,2,5)), lon=lon, lat=lat, time=time)  # lat x lon x time
    cube.axes  # doctest: +ELLIPSIS



.. parsed-literal::

    dimensions: 'lat', 'lon', 'time'
    0 / lat (3): 10.0 to 50.0
    1 / lon (2): 10.0 to 30.0
    2 / time (5): 1950 to 1954



.. code:: python

    ### In combination with repeat
    a.newaxis('new', values=['a','b'],pos=1) # repeat 2 times along the first axis



.. parsed-literal::

    dimarray: 4 non-null elements (0 null)
    dimensions: 'lon', 'new'
    0 / lon (2): 30.0 to 40.0
    1 / new (2): a to b
    array([[0, 0],
           [1, 1]])



Broadcast
~~~~~~~~~


.. code:: python

    # broadcast along new axes (reshape + repeat array)
    >>> ts3D = ts.broadcast(cube) #  lat x lon x time
    ts3D



.. parsed-literal::

    dimarray: 30 non-null elements (0 null)
    dimensions: 'lat', 'lon', 'time'
    0 / lat (3): 10.0 to 50.0
    1 / lon (2): 10.0 to 30.0
    2 / time (5): 1950 to 1954
    array([[[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]],
    
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]],
    
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])



Align arrays
------------


Align dimensions
~~~~~~~~~~~~~~~~


.. code:: python

    # Keep new axes as singleton dimensions
    x = da.DimArray(np.arange(2), dims=('x0',))
    y = da.DimArray(np.arange(3), dims=('x1',))
    da.align_dims(x, y)



.. parsed-literal::

    [dimarray: 2 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (1): None to None
    array([[0],
           [1]]),
     dimarray: 3 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (1): None to None
    1 / x1 (3): 0 to 2
    array([[0, 1, 2]])]



.. code:: python

    # Broadcast arrays: all array have same size
    da.broadcast_arrays(x, y)



.. parsed-literal::

    [dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[0, 0, 0],
           [1, 1, 1]]),
     dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[0, 1, 2],
           [0, 1, 2]])]



Align axes
~~~~~~~~~~


.. code:: python

    a = da.DimArray([1,2,3],('x0',[1,2,3]))
    b = da.DimArray([3,4],('x0',[2,4]))
    da.align_axes(a, b)



.. parsed-literal::

    [dimarray: 3 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (4): 1 to 4
    array([  1.,   2.,   3.,  nan]),
     dimarray: 2 non-null elements (2 null)
    dimensions: 'x0'
    0 / x0 (4): 1 to 4
    array([ nan,   3.,  nan,   4.])]



Join arrays
-----------


Concatenate arrays along existing axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: python

    import dimarray as da
    import numpy as np
    a = da.DimArray([[1.,2,3]],axes=[('line',[1]), ('col',['a','b','c'])])
    b = da.DimArray([[4,5,6],[7,8,9]], axes=[('line',[2,3]), ('col',['a','b','c'])])
    da.concatenate((a,b))



.. parsed-literal::

    dimarray: 9 non-null elements (0 null)
    dimensions: 'line', 'col'
    0 / line (3): 1 to 3
    1 / col (3): a to c
    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.],
           [ 7.,  8.,  9.]])



From a collection of DimArray objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: python

    from dimarray import DimArray
    a = DimArray([1,2,3])
    DimArray.from_arrays({'a':a, '2*a':2*a}) 



.. parsed-literal::

    dimarray: 6 non-null elements (0 null)
    dimensions: 'items', 'x0'
    0 / items (2): a to 2*a
    1 / x0 (3): 0 to 2
    array([[1, 2, 3],
           [2, 4, 6]])



or with a list

.. code:: python

    DimArray.from_arrays([a, 2*a], keys=['a','2*a']) # keys would be v0 and v1 by default



.. parsed-literal::

    dimarray: 6 non-null elements (0 null)
    dimensions: 'items', 'x0'
    0 / items (2): a to 2*a
    1 / x0 (3): 0 to 2
    array([[1, 2, 3],
           [2, 4, 6]])



this allows on the fly axis alignment:

.. code:: python

    d = {'a':DimArray([10,20,30.],labels=[0,1,2]), 'b':DimArray([1,2,3.],labels=[1.,2.,3])}
    a = DimArray.from_arrays(d, keys=['a','b']) # keys= just needed to enforce ordering
    a



.. parsed-literal::

    dimarray: 6 non-null elements (2 null)
    dimensions: 'items', 'x0'
    0 / items (2): a to b
    1 / x0 (4): 0 to 3
    array([[ 10.,  20.,  30.,  nan],
           [ nan,   1.,   2.,   3.]])



Works in any number of dimensions

.. code:: python

    d = {'a':DimArray([[10,20,30.],[0,1,2]]), 'b':DimArray([[1,2,3.],[1.,2.,3]])}
    a = DimArray.from_arrays(d, keys=['a','b']) # keys= just needed to enforce ordering
    a



.. parsed-literal::

    dimarray: 12 non-null elements (0 null)
    dimensions: 'items', 'x0', 'x1'
    0 / items (2): a to b
    1 / x0 (2): 0 to 1
    2 / x1 (3): 0 to 2
    array([[[ 10.,  20.,  30.],
            [  0.,   1.,   2.]],
    
           [[  1.,   2.,   3.],
            [  1.,   2.,   3.]]])



Aggregate arrays of varying dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: python

    %pylab
    a = arange(4)
    a[(a==2)+0]
    (a==2)+0
    a[[0,1,0,1]]
    #a[np.array([0,1,0,1])]
    a[np.array([0,1,0,1], dtype=int) == True]

.. parsed-literal::

    Using matplotlib backend: GTKAgg
    Populating the interactive namespace from numpy and matplotlib


.. parsed-literal::

    WARNING: pylab import has clobbered these variables: ['e']
    `%pylab --no-import-all` prevents importing * from pylab and numpy




.. parsed-literal::

    array([1, 3])



Here a nice example of puzzle filling (values in the output array
indicate the order of insertion):

.. code:: python

    import dimarray as da
    import numpy as np
    a = da.DimArray([[1.,2,3]],axes=[('line',[1]), ('col',['a','b','c'])])
    b = da.DimArray([[4],[5]], axes=[('line',[2,3]), ('col',['d'])])
    c = da.DimArray([[6]], axes=[('line',[2]), ('col',['b'])])
    d = da.DimArray([-7], axes=[('line',[4])])
    da.aggregate((a,b,c,d))



.. parsed-literal::

    dimarray: 10 non-null elements (6 null)
    dimensions: 'line', 'col'
    0 / line (4): 1 to 4
    1 / col (4): a to d
    array([[  1.,   2.,   3.,  nan],
           [ nan,   6.,  nan,   4.],
           [ nan,  nan,  nan,   5.],
           [ -7.,  -7.,  -7.,  -7.]])



Risk of overlapping checked. In case of overlapping of a valid and an
invalid value, keep the valid one

.. code:: python

    a = da.DimArray([[1.,2,3]],axes=[('line',[1]), ('col',['a','b','c'])])
    e = da.DimArray([[np.nan],[5]], axes=[('line',[1,2]), ('col',['b'])])
    da.aggregate((a,e)) # does not overwrite `2` at location (1, 'b')



.. parsed-literal::

    dimarray: 4 non-null elements (2 null)
    dimensions: 'line', 'col'
    0 / line (2): 1 to 2
    1 / col (3): a to c
    array([[  1.,   2.,   3.],
           [ nan,   5.,  nan]])



But any loss of data (overlap between two valid values) is prevented by
raising an exception:

.. code:: python

    a = da.DimArray([[1.,2,3]],axes=[('line',[1]), ('col',['a','b','c'])])
    e = da.DimArray([[4],[5]], axes=[('line',[1,2]), ('col',['b'])])
    try:
        da.aggregate((a,e))
    except Exception, msg:
        print msg

.. parsed-literal::

    Overlapping arrays: set check_overlap to False to suppress this error.


Unless specified otherwise with ``check_overlap=False`` (will also
speedup the operation)

.. code:: python

    da.aggregate((a,e), check_overlap=False)



.. parsed-literal::

    dimarray: 4 non-null elements (2 null)
    dimensions: 'line', 'col'
    0 / line (2): 1 to 2
    1 / col (3): a to c
    array([[  1.,   4.,   3.],
           [ nan,   5.,  nan]])



Operations
----------


Basic Operations
~~~~~~~~~~~~~~~~


.. code:: python

    a = da.DimArray([[1,2,3],[3,4,5]],dims=('x0','x1'))
    assert np.all(a == a)
    assert np.all(a+2 == a + np.ones(a.shape)*2)
    assert np.all(a+a == a*2)
    assert np.all(a*a == a**2)
    assert np.all((a - a.values) == a - a)
    a == a



.. parsed-literal::

    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[ True,  True,  True],
           [ True,  True,  True]], dtype=bool)



Operation with data alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: python

    import dimarray as da
    import numpy as np
    test = da.DimArray([1, 2]) == 1
    test2 = da.DimArray([1, 2]) == da.DimArray([1, 2])
    test3 = da.DimArray([1, 2]) == np.array([1, 2])

.. code:: python

    # broadcasting
    x = da.DimArray(np.arange(2), dims=('x0',))
    y = da.DimArray(np.arange(3), dims=('x1',))
    x+y



.. parsed-literal::

    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[0, 1, 2],
           [1, 2, 3]])



.. code:: python

    # axis alignment
    z = da.DimArray([0,1,2],('x0',[0,1,2]))
    x+z



.. parsed-literal::

    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  0.,   2.,  nan])



.. code:: python

    # or both
    (x+y)+(x+z)



.. parsed-literal::

    dimarray: 6 non-null elements (3 null)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (3): 0 to 2
    array([[  0.,   1.,   2.],
           [  3.,   4.,   5.],
           [ nan,  nan,  nan]])



Numpy Transformations
---------------------


.. code:: python

    np.random.seed(0)
    v = da.DimArray(np.random.randn(5,7), {'time':np.arange(1950,1955), 'lat':np.linspace(-90,90,7)})
Basic transforms: reduce axis

.. code:: python

    v.sum() # sum over all axes
    v.sum(axis=0) # sum over first axis
    v.sum(axis='time') # named axis
    v.prod(axis='time') # named axis
    v.mean(axis='time') # named axis
    v.var(axis='time') # named axis
    v.std(axis='time') # named axis
    v.median(axis='time') # named axis
    v.min(axis='time') # named axis
    v.max(axis='time') # named axis
    v.ptp(axis='time') # named axis
    v.all(axis='time') # named axis
    v.any(axis='time') # named axis



.. parsed-literal::

    dimarray: 7 non-null elements (0 null)
    dimensions: 'lat'
    0 / lat (7): -90.0 to 90.0
    array([ True,  True,  True,  True,  True,  True,  True], dtype=bool)



locmin/locmax as dimarray equivalent of argmin/argmax: return axis value

.. code:: python

    v.locmin(axis='time') 
    v.locmin() # over all axes
    v.locmax(axis='time') 
    v.locmax() # over all axes
    v.locmax(axis='time')
    assert v[v.locmax()] == v.max(), "problem with locmax/max"
    assert v[v.locmin()] == v.min(), "problem with locmin/min"
    res = np.array([v.ix[i][ix] for i, ix in enumerate(v.locmax(axis=1).values)])
    assert np.all(res == v.max(axis=1)), "problem with locmax/max"
    v.locmax()



.. parsed-literal::

    (1953, 0.0)



cumulative transformation

.. code:: python

    v.cumsum() # last axis
    v.cumsum(axis=None) # return numpy array
    assert np.all(v.cumsum(axis=-1) == v.cumsum()), "default last axis"
new diff method

.. code:: python

    v.diff(axis='time', keepaxis=False)
    v.diff(axis=0, keepaxis=False, scheme='centered')
    v.diff(axis=0, keepaxis=False, scheme='backward')
    v.diff(axis=0, keepaxis=False, scheme='forward')
    v.diff(axis=0, keepaxis=True, scheme='backward')
    v.diff(axis=0, keepaxis=True, scheme='forward')
    v.diff(n=2,axis=('time'), scheme='centered')



.. parsed-literal::

    dimarray: 21 non-null elements (0 null)
    dimensions: 'time', 'lat'
    0 / time (3): 1951.0 to 1953.0
    1 / lat (7): -90.0 to 90.0
    array([[ 2.51063   ,  0.94026924,  1.65162005,  1.74764779, -0.72792132,
            -3.35344907, -1.84625143],
           [-0.38546508,  0.09386869, -3.31972466,  2.82411472, -0.62622757,
             2.51498772,  5.0404708 ],
           [ 0.66940526,  0.0741607 ,  3.13335654, -4.36650499,  2.3340133 ,
            -2.92640924, -2.52653427]])



.. code:: python

    # CHECK AGAINST PANDAS
    assert np.all(v.std(ddof=1, axis=0).values==v.to_pandas().std().values), "std vs pandas"
    assert np.sum((v.var(ddof=1, axis=0).values-v.to_pandas().var().values)**2)<1e-10, "var vs pandas"
    assert np.all(v.cumsum(axis=0).values == v.to_pandas().cumsum().values), "error against pandas"
    assert np.all(v.cumprod(axis=0).values == v.to_pandas().cumprod().values), "error against pandas"
    assert np.nansum((v.diff(axis=0, keepaxis=True).cumsum(axis=0, skipna=True).values - v.to_pandas().diff().cumsum().values)**2) \
      < 1e-10, "error against pandas"
Dataset
-------


Aggregate several DimArray to a Dataset object, it will automatically
align the data.

It can be empty, just given some axes:

.. code:: python

    import dimarray as da
    axes = da.Axes.from_tuples(('time',[1, 2, 3]))
    ds = da.Dataset(axes=axes)
    a = da.DimArray([[0, 1],[2, 3]], dims=('time','items'))
    ds['yo'] = a.reindex_like(ds)
.. code:: python

    import dimarray as da
    %pylab
    seed(0)
    mymap = da.DimArray.from_kw(randn(50,7), lon=linspace(-180,180,50), lat=linspace(-90,90,7))
    ts = da.DimArray(np.arange(5), ('time',arange(1950,1955)))
    ts2 = da.DimArray(np.arange(10), ('time',arange(1950,1960)))
    
    # Define a Dataset made of several variables
    data = da.Dataset({'ts':ts, 'ts2':ts2, 'mymap':mymap})
    data = da.Dataset([ts, ts2, mymap], keys=['ts','ts2','mymap'])

.. parsed-literal::

    Using matplotlib backend: GTKAgg
    Populating the interactive namespace from numpy and matplotlib


.. parsed-literal::

    WARNING: pylab import has clobbered these variables: ['axes', 'test', 'e']
    `%pylab --no-import-all` prevents importing * from pylab and numpy


Data have been automatically aligned, while keeping the same shape

.. code:: python

    assert np.all(data['ts'].time == data['ts2'].time),"Dataset: pb data alignment" 
    data['ts']



.. parsed-literal::

    dimarray: 5 non-null elements (5 null)
    dimensions: 'time'
    0 / time (10): 1950 to 1959
    array([  0.,   1.,   2.,   3.,   4.,  nan,  nan,  nan,  nan,  nan])



Can also add any other data as long as it is aligned with the dataset

.. code:: python

    data['test'] = da.DimArray([0],('source',['greenland']))  # new axis
    try:
        data['test2'] = da.DimArray([0,3],('source',['greenland','antarctica']))
    except Exception, msg:
        print msg
    data
    ## TODO: "expand" Dataset (such as reindex_axis)

.. parsed-literal::

    axes values do not match, align data first.			    
    Dataset: source(1)=greenland:greenland, 
    Got: source(2)=greenland:antarctica




.. parsed-literal::

    Dataset of 4 variables
    dimensions: 'time', 'lon', 'lat', 'source'
    0 / time (10): 1950 to 1959
    1 / lon (50): -180.0 to 180.0
    2 / lat (7): -90.0 to 90.0
    3 / source (1): greenland to greenland
    ts: ('time',)
    ts2: ('time',)
    mymap: ('lon', 'lat')
    test: ('source',)



.. code:: python

    # Export to a DimArray
    print data.to_array()

.. parsed-literal::

    dimarray: 12250 non-null elements (1750 null)
    dimensions: 'items', 'time', 'lon', 'lat', 'source'
    0 / items (4): ts to test
    1 / time (10): 1950 to 1959
    2 / lon (50): -180.0 to 180.0
    3 / lat (7): -90.0 to 90.0
    4 / source (1): greenland to greenland
    array(...)


NetCDF I/O
----------


.. code:: python

    from dimarray import DimArray, summary_nc, read_nc
    a = DimArray([1,2], dims='xx0')
    b = DimArray([3,4,5], dims='xx1')
    a.write_nc("test.nc","a", mode='w')
    b.write_nc("test.nc","b", mode='a')
    data = read_nc("test.nc")
    data

.. parsed-literal::

    read from test.nc




.. parsed-literal::

    Dataset of 2 variables
    dimensions: 'xx0', 'xx1'
    0 / xx0 (2): 0 to 1
    1 / xx1 (3): 0 to 2
    a: ('xx0',)
    b: ('xx1',)



A real-data example

.. code:: python

    path="/media/Data/Data/All/Etopo/Etopo5.cdf"
    import dimarray.io.nc as ncio; reload(ncio) 
    import dimarray as da
    print da.summary_nc(path)
    da.read_nc(path, "elev", indices={"X":slice(0.0,10), "Y":slice(80., 70)}, tol=0.1)
    #da.read_nc(path, "elev", indices={"X":slice(10), "Y":slice(-10,None)}, numpy_indexing=True)

.. parsed-literal::

    /media/Data/Data/All/Etopo/Etopo5.cdf:
    -------------------------------------
    Dataset of 1 variable
    dimensions: 'Y', 'X'
    0 / Y (2160): 90.0 to -89.9166641235
    1 / X (4320): 0.0 to 359.916656494
    elev: (u'Y', u'X')
    None
    read from /media/Data/Data/All/Etopo/Etopo5.cdf




.. parsed-literal::

    dimarray: 14641 non-null elements (0 null)
    dimensions: 'Y', 'X'
    0 / Y (121): 80.0 to 70.0
    1 / X (121): 0.0 to 10.0
    array(...)



From a DimArray
~~~~~~~~~~~~~~~


.. code:: python

    seed(0)
    v = da.DimArray(randn(5,7,2), [("time",np.arange(1950,1955)), ("lat",np.linspace(-90,90,7)), ("items",np.array(['greenland','antarctica']))])
    v



.. parsed-literal::

    dimarray: 70 non-null elements (0 null)
    dimensions: 'time', 'lat', 'items'
    0 / time (5): 1950 to 1954
    1 / lat (7): -90.0 to 90.0
    2 / items (2): greenland to antarctica
    array([[[ 1.76405235,  0.40015721],
            [ 0.97873798,  2.2408932 ],
            [ 1.86755799, -0.97727788],
            [ 0.95008842, -0.15135721],
            [-0.10321885,  0.4105985 ],
            [ 0.14404357,  1.45427351],
            [ 0.76103773,  0.12167502]],
    
           [[ 0.44386323,  0.33367433],
            [ 1.49407907, -0.20515826],
            [ 0.3130677 , -0.85409574],
            [-2.55298982,  0.6536186 ],
            [ 0.8644362 , -0.74216502],
            [ 2.26975462, -1.45436567],
            [ 0.04575852, -0.18718385]],
    
           [[ 1.53277921,  1.46935877],
            [ 0.15494743,  0.37816252],
            [-0.88778575, -1.98079647],
            [-0.34791215,  0.15634897],
            [ 1.23029068,  1.20237985],
            [-0.38732682, -0.30230275],
            [-1.04855297, -1.42001794]],
    
           [[-1.70627019,  1.9507754 ],
            [-0.50965218, -0.4380743 ],
            [-1.25279536,  0.77749036],
            [-1.61389785, -0.21274028],
            [-0.89546656,  0.3869025 ],
            [-0.51080514, -1.18063218],
            [-0.02818223,  0.42833187]],
    
           [[ 0.06651722,  0.3024719 ],
            [-0.63432209, -0.36274117],
            [-0.67246045, -0.35955316],
            [-0.81314628, -1.7262826 ],
            [ 0.17742614, -0.40178094],
            [-1.63019835,  0.46278226],
            [-0.90729836,  0.0519454 ]]])



.. code:: python

    # writing
    v.write_nc("test2.nc","myvarstr", mode="w")
.. code:: python

    # checking
    da.summary_nc("test2.nc")

.. parsed-literal::

    test2.nc:
    --------
    Dataset of 1 variable
    dimensions: 'time', 'lat', 'items'
    0 / time (5): 1950 to 1954
    1 / lat (7): -90.0 to 90.0
    2 / items (2): greenland to antarctica
    myvarstr: (u'time', u'lat', u'items')


.. code:: python

    # reading
    w = da.read_nc("test2.nc")
    assert np.all(w['myvarstr'] == v), "Problem when reading netcdf"
    w

.. parsed-literal::

    read from test2.nc




.. parsed-literal::

    Dataset of 1 variable
    dimensions: 'time', 'lat', 'items'
    0 / time (5): 1950 to 1954
    1 / lat (7): -90.0 to 90.0
    2 / items (2): greenland to antarctica
    myvarstr: ('time', 'lat', 'items')



From/To Dataset
~~~~~~~~~~~~~~~


.. code:: python

    data.write('test3.nc','w')
    da.summary_nc('test3.nc')

.. parsed-literal::

    write to test3.nc
    test3.nc:
    --------
    Dataset of 2 variables
    dimensions: 'xx0', 'xx1'
    0 / xx0 (2): 0 to 1
    1 / xx1 (3): 0 to 2
    a: (u'xx0',)
    b: (u'xx1',)


Experimental Features
---------------------


Group / Ungroup
~~~~~~~~~~~~~~~


.. code:: python

    %pylab
    import dimarray as da
    np.random.seed(0)
    v = da.DimArray.from_kw(np.random.randn(2,5,180), time=np.arange(1950,1955), lat=np.linspace(-90,90,180), items=np.array(['greenland','antarctica']))
    v

.. parsed-literal::

    Using matplotlib backend: GTKAgg
    Populating the interactive namespace from numpy and matplotlib




.. parsed-literal::

    dimarray: 1800 non-null elements (0 null)
    dimensions: 'items', 'time', 'lat'
    0 / items (2): greenland to antarctica
    1 / time (5): 1950 to 1954
    2 / lat (180): -90.0 to 90.0
    array(...)



Flatten a set of dimensions:

.. code:: python

    w = v.group(('time','lat'))
    w



.. parsed-literal::

    dimarray: 1800 non-null elements (0 null)
    dimensions: 'time,lat', 'items'
    0 / time,lat (900): (1950, -90.0) to (1954, 90.0)
    1 / items (2): greenland to antarctica
    array(...)



Ungroup: get back to previous full n-d array

.. code:: python

    w.ungroup()



.. parsed-literal::

    dimarray: 1800 non-null elements (0 null)
    dimensions: 'time', 'lat', 'items'
    0 / time (5): 1950 to 1954
    1 / lat (180): -90.0 to 90.0
    2 / items (2): greenland to antarctica
    array(...)



pass a tuple or list of dimensions to an axis-transform, to pre-flatten
grouping in transformations

.. code:: python

    v.mean(axis=('time','lat'))



.. parsed-literal::

    dimarray: 2 non-null elements (0 null)
    dimensions: 'items'
    0 / items (2): greenland to antarctica
    array([-0.06849458,  0.04075616])



Weighted mean
~~~~~~~~~~~~~


Each axis can have a ``weights`` attribute. If not None, it will be
automatically used when computing mean, var, std

.. code:: python

    >>> np.random.seed(0)
    >>> data = da.DimArray(randn(7), ('lat',linspace(-90,90,7)))
    
    >>> umean = data.mean(axis='lat')
    print "\nunweighted zonal mean:\n\n", umean
    
    # now add weights
    >>> data.axes['lat'].weights = lambda x: np.cos(np.radians(x))
    
    >>> wmean = data.mean(axis='lat')
    print "\nweighted zonal mean:\n", wmean
    
    >>> wmean = data[:].mean(axis='lat')
    print "\nweighted zonal mean after [:]:\n", wmean
    
    >>> test =  data.mean(axis='lat', weights=None) 
    print "\nCHECK:", test == umean

.. parsed-literal::

    
    unweighted zonal mean:
    
    1.03202989506
    
    weighted zonal mean:
    1.18361129352
    
    weighted zonal mean after [:]:
    1.18361129352
    
    CHECK: True


.. code:: python

    # Weighted mean with grouping
    #>>> mymap[mymap.lon > 0])
    #lon=linspace(-180,180,50), 
    mymap = data.reshape(('lat','lon')).repeat(linspace(-180,180,5), axis='lon')
    mymap.values[:,0] = np.nan
    #>>> print mymap
    #lon=linspace(-180,180,50)
    wmean = mymap.mean()
    print "\nNaN + flatten all:\n",umean
    #>>> wmean = mymap.mean(axis=('lon','lat'))
    wmean = mymap.mean(axis='lon').mean(axis='lat')
    print "\nNaN + sequential:\n",wmean
    cube = da.DimArray(randn(50,7,5), [('lon',linspace(-180,180,50)), ("lat",linspace(-90,90,7)), ('time',arange(1950,1955))])
    cube.axes['lat'].weights = lambda x: np.cos(np.radians(x))

.. parsed-literal::

    
    NaN + flatten all:
    1.03202989506
    
    NaN + sequential:
    nan


Interpolation
~~~~~~~~~~~~~


.. code:: python

    time=np.linspace(1950,1955,8)
    v = da.DimArray.from_kw(cos(time), time=time)
    #w = v.reindex_axis(arange(1940,1960), axis='time') 
    w = da.interp1d(v, np.linspace(1948,1957,10), axis='time')
    #%matplotlib inline
    clf()
    plot(v.time, v.values, label='original')
    plot(w.time, w.values, label='interp')
    legend()
    show()
    draw()
Export to other formats
~~~~~~~~~~~~~~~~~~~~~~~


.. code:: python

    #reload(da) # for dict
    v = da.DimArray.from_kw(randn(2,5,7), time=arange(1950,1955), lat=linspace(-90,90,7), items=array(['greenland','antarctica']))
    #reload(co) # for dict
    
    print "\nExport to pandas\n"
    print v.to_pandas()
    print "\nExport to larry\n"
    print v.to_larry()
    #print "\nExport to Dataset\n"
    #print v.to()
    #v.to_MaskedArray() # to masked array

.. parsed-literal::

    
    Export to pandas
    
    <class 'pandas.core.panel.Panel'>
    Dimensions: 2 (items) x 5 (major_axis) x 7 (minor_axis)
    Items axis: greenland to antarctica
    Major_axis axis: 1950 to 1954
    Minor_axis axis: -90.0 to 90.0
    
    Export to larry
    
    warning: dimension names have not been passed to larry
    label_0
        greenland
        antarctica
    label_1
        1950
        1951
        1952
        1953
        1954
    label_2
        -90.0
        -60.0
        -30.0
        0.0
        30.0
        60.0
        90.0
    x
    array([[[ 0.02429091,  1.27981202, -0.88596648,  0.40088568, -0.00965724,
             -1.79716462, -0.80225317],
            [ 0.19321355,  1.29734209,  1.00133102,  0.5972125 , -0.81527566,
              1.80121399,  0.21524047],
            [-1.00636552, -0.18290498,  0.89624843,  0.0076175 ,  0.88686469,
              1.10369396,  0.40053068],
            [-0.85770262,  0.13545466,  0.04516586,  1.85934633, -1.62632194,
             -0.13482245, -0.58409355],
            [ 0.33510562, -2.43756436,  1.11492456,  0.01374849, -1.84470116,
             -0.36111313,  0.60896234]],
    
           [[-1.59144788,  0.00322222, -1.05747365, -0.55598503,  0.02673838,
              0.18345025, -0.4707425 ],
            [ 0.27279639,  0.81797761, -0.27891428,  1.43156776,  1.46221417,
             -0.42870207, -0.63784056],
            [-1.66417299, -0.12656933, -0.36343778,  0.77905122, -1.50966161,
             -0.27739139,  0.96874439],
            [-0.7303571 , -0.76236154, -1.44694033,  2.62057385, -0.74747318,
             -1.30034683, -0.8038504 ],
            [-0.77429508, -0.26938978,  0.82537223, -0.29832317, -0.92282331,
             -1.4513385 ,  0.02185736]]])


.. code:: python

    >>> from pandas import Series, DataFrame
    >>> s = Series([3,5,6], index=['a','b','c'])
    >>> s.index.name = 'dim0'
    >>> print DimArray.from_pandas(s)
    >>> d = DataFrame([[3,5,6],[1,2,3]], index=[50,20], columns=['a','b','c'])
    >>> d.index.name = 'ii'
    >>> d.columns.name = 'cc'
    >>> print 
    >>> print DimArray.from_pandas(d)

.. parsed-literal::

    dimarray: 3 non-null elements (0 null)
    dimensions: 'dim0'
    0 / dim0 (3): a to c
    array([3, 5, 6], dtype=int64)
    
    dimarray: 6 non-null elements (0 null)
    dimensions: 'ii', 'cc'
    0 / ii (2): 50 to 20
    1 / cc (3): a to c
    array([[3, 5, 6],
           [1, 2, 3]], dtype=int64)


doctest framework
-----------------


.. code:: python

    import dimarray.tests as tests
    #import dimarray.tests as tests
    tests.main()
    #run test.test_all()

.. parsed-literal::

    
    
    ============================
    TEST dimarray.core.metadata
    ============================
    
    
    
    
    ============================
    TEST dimarray.core.dimarraycls
    ============================
    
    
    
    
    ============================
    TEST dimarray.core.axes
    ============================
    
    
    
    
    ============================
    TEST dimarray.core._indexing
    ============================
    
    
    
    
    ============================
    TEST dimarray.core._transform
    ============================
    
    
    
    
    ============================
    TEST dimarray.core._reshape
    ============================
    
    
    
    
    ============================
    TEST dimarray.core.missingvalues
    ============================
    
    
    
    
    ============================
    TEST dimarray.core._operation
    ============================
    
    
    
    
    ============================
    TEST dimarray.core.align
    ============================
    
    
    
    
    ============================
    TEST dimarray.core.tests
    ============================
    
    
    
    
    ============================
    TEST dimarray.geo.geoarray
    ============================
    
    
    
    
    ============================
    TEST dimarray.geo.region
    ============================
    
    
    
    
    ============================
    TEST dimarray.geo.transform
    ============================
    
    
    
    
    ============================
    TEST dimarray.geo.grid
    ============================
    
    
    
    
    ============================
    TEST decorator
    ============================
    
    
    
    
    ============================
    TEST dimarray
    ============================
    
    
    
    
    ============================
    TEST dimarray.dataset
    ============================
    
    

