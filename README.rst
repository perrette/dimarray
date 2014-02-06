dimarray: array with labelled dimensions 
========================================

Idea:
-----
Have a numpy array with labelled axes, like `larry` or `pandas`, 
with additional focus on axis names (the dimensions, without restriction on their number). 
This means that most standard transforms (e.g. `transpose`, `mean` etc...) 
take axis name for the `axis=` parameter, in addition to its integer position.

Having axis name and axis values allow on-the-fly axis alignment and broadcasting 
in basic operations (addition, etc...), so that rules can be defined for nearly 
every sequence of operands. 

Indexing works on axis values by default, with a `ix` toogle to switch to indexing
on position like numpy, and generic `take`/`put` methods.

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


Notebook tutorial:
------------------
http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb

Further development:
--------------------
All suggestions for improvement very welcome, please file an `issue` on github:
https://github.com/perrette/dimarray/ for further discussion.
