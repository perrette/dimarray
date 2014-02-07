dimarray: array with labelled dimensions 
========================================

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


Get started
-----------

A **``DimArray``** can be defined just like a numpy array, with
additional information about its dimensions, which can be given 
via `labels` and `dims` parameters.

.. code:: python

    from dimarray import DimArray

.. code:: python

    a = DimArray([[1,2,3],[4,5,6]], labels = [["a","b"], [0,1,2]], dims=["dim0","dim1"])
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


And axis names and labels can be accessed straightforwardly:


.. code:: python

    a.dims 



.. parsed-literal::

    ('dim0', 'dim1')



.. code:: python

    a.labels


.. parsed-literal::

    (array(['a', 'b'], dtype=object), array([0, 1, 2]))


Indexing works on labels just as expected, including `slice` and boolean array. 

.. code:: python

    a['b', 1]

.. parsed-literal::

    5

but integer-index is always possible via `ix` toogle between `labels`- and `position`-based indexing:

.. code:: python

    a.ix[1, 1]

.. parsed-literal::

    5


Numpy transformations are defined, and now accept axis name:


.. code:: python

    a.mean(axis='dim0')

.. parsed-literal::

    array([2.5, 3.5, 4.5])


and can ignore NaNs if asked to:

.. code:: python

    a['a',2] = nan
    a.mean(axis='dim0', skipna=True)

.. parsed-literal::

    array([2.5, 3.5, 6.])



More on the notebook documentation:
http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb


Comparisons with other packages:
--------------------------------

- `pandas` is an excellent package for low-dimensional data analysis, 
    with many I/O features, but is mostly limited to 2 dimensions
    (DataFrame), or up to 4 dimensions (Panel, Panel4D). `dimarray` includes
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

- `larry` was pioneer as labelled array, it skips nans in along-axis transforms
    and comes with many handy methods. After giving it a go, I find it is not 
    so intuitive to use, but this is a matter of taste. `larry` does not seem 
    to support naming dimensions.

Compared with these two pacakges, `dimarray` adds the possibility of passing axis 
name to the various methods, instead of simply axis rank. This applies for 
instance to along-axis operation, `take` and `put` methods, or reshaping operations.

- `iris` looks like a very powerful package to manipulate geospatial data with 
    metadata, netCDF I/O, performing grid transforms etc..., but it is quite a jump 
    from numpy's `ndarray` and requires a bit of learning. 
    In contrast, `dimarray` is more general and intuitive for python users. `dimarray`
    also comes with netCDF I/O capability and may gain a few geospatial features 
    (weighted mean for lon/lat, 360 modulo for lon, regridding, etc...) as a subpackage 
    dimarray.geo -- and why not an interface to `iris`.


Further development:
--------------------
All suggestions for improvement very welcome, please file an `issue` on github:
https://github.com/perrette/dimarray/ for further discussion.
