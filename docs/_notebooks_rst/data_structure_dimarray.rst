.. This file was generated automatically from the ipython notebook:
.. notebooks/data_structure_dimarray.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_data_structure_dimarray:


.. _DimArray_class:

DimArray class
--------------
:download:`Download notebook </notebooks/data_structure_dimarray.ipynb>` 


Let's use this dimarray as example:

>>> from dimarray import DimArray
>>> a = DimArray([[1.,2,3], [4,5,6]], axes=[['a', 'b'], [1950, 1960, 1970]], dims=['variable', 'time'])
>>> a
dimarray: 6 non-null elements (0 null)
0 / variable (2): 'a' to 'b'
1 / time (3): 1950 to 1970
array([[1., 2., 3.],
       [4., 5., 6.]])

.. _values_and_axes:

values and axes
^^^^^^^^^^^^^^^

Array data are stored in a `values` **attribute**:

>>> a.values
array([[1., 2., 3.],
       [4., 5., 6.]])

while its axes are stored in `axes`:

>>> a.axes
0 / variable (2): 'a' to 'b'
1 / time (3): 1950 to 1970

An axis is the equivalent of pandas's index, except that it always has a name. Each axis can be accessed by its rank or its name:

>>> ax0 = a.axes[0]         # "variable" axis by rank 
>>> ax0
variable (2): 'a' to 'b'

>>> ax1 = a.axes['time']    # "time" axis by name
>>> ax1
time (3): 1950 to 1970

Name and values can be accessed as expected.

>>> ax1.name
'time'

>>> ax1.values
array([1950, 1960, 1970])

In many cases (like plotting) you just want the values, so for convenience you can just access them as a DimArray attribute via axis name (as long as the name is not a protected DimArray attribute):

>>> a.time
array([1950, 1960, 1970])

.. _numpy-like_attributes:

numpy-like attributes
^^^^^^^^^^^^^^^^^^^^^

Numpy-like attributes `dtype`, `shape`, `size` or `ndim` are defined, and are now augmented with `dims` and `labels`

>>> a.shape
(2, 3)

>>> a.dims      # grab axis names (the dimensions)
('variable', 'time')

>>> a.labels   # grab axis values
(array(['a', 'b'], dtype=object), array([1950, 1960, 1970]))

while dimensions are located in an `axes` attribute, an `Axes` instance

>>> a.axes
0 / variable (2): 'a' to 'b'
1 / time (3): 1950 to 1970

Individual axes can be accessed as well, which are `Axis` instances.

.. _metadata:

metadata
^^^^^^^^

The straightforward way to define them is via the standard `.` syntax to access an object attribute:

>>> from dimarray import DimArray
>>> a = DimArray([1,2,3])


>>> a.name = 'myname'
>>> a.units = 'myunits'


All metadata are stored in an `attrs` dictionary attribute (via overloading __setattr__):

>>> a.attrs
OrderedDict([('name', 'myname'), ('units', 'myunits')])

Metadata are conserved by slicing and along-axis transformation, but are lost with any other transformation.

>>> a[:].attrs
OrderedDict([('name', 'myname'), ('units', 'myunits')])

.. note:: For metadata that could conflict with protected attributes, or with axis names, please use `attrs` directly to set or get the metadata.