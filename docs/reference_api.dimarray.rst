.. _ref_api_methods:

==============================
DimArray methods reference API
==============================

DimArray methods are list below by topic, along with examples. 
Functions are provided in a separate page :ref:`ref_api_functions`.

.. contents:: 
    :local:
    :depth: 2


Create a DimArray
-----------------

.. automethod:: dimarray.DimArray.__init__

-------------------------

.. automethod:: dimarray.DimArray.from_kw


.. _ref_api_reshaping:

Modify shape
------------

.. automethod:: dimarray.DimArray.transpose

-------------------------

.. automethod:: dimarray.DimArray.swapaxes

-------------------------

.. automethod:: dimarray.DimArray.reshape

-------------------------

.. automethod:: dimarray.DimArray.group

-------------------------

.. automethod:: dimarray.DimArray.ungroup

-------------------------

.. automethod:: dimarray.DimArray.flatten

-------------------------

.. automethod:: dimarray.DimArray.newaxis

-------------------------

.. automethod:: dimarray.DimArray.squeeze

-------------------------

.. automethod:: dimarray.DimArray.repeat

-------------------------

.. automethod:: dimarray.DimArray.broadcast

Reduce, accumulate
------------------

.. automethod:: dimarray.DimArray.max
   
-------------------------

.. automethod:: dimarray.DimArray.min

-------------------------

.. automethod:: dimarray.DimArray.ptp

-------------------------

.. automethod:: dimarray.DimArray.median

-------------------------

.. automethod:: dimarray.DimArray.all

-------------------------

.. automethod:: dimarray.DimArray.any

-------------------------

.. automethod:: dimarray.DimArray.prod

-------------------------

.. automethod:: dimarray.DimArray.sum
 
-------------------------

.. automethod:: dimarray.DimArray.mean

-------------------------

.. automethod:: dimarray.DimArray.std

-------------------------

.. automethod:: dimarray.DimArray.var

-------------------------

.. automethod:: dimarray.DimArray.argmax

-------------------------

.. automethod:: dimarray.DimArray.argmin

-------------------------

.. automethod:: dimarray.DimArray.cumsum

-------------------------

.. automethod:: dimarray.DimArray.cumprod

-------------------------

.. automethod:: dimarray.DimArray.diff


Indexing
--------

.. automethod:: dimarray.DimArray.__getitem__

-------------------------

.. automethod:: dimarray.DimArray.ix

-------------------------

.. automethod:: dimarray.DimArray.box

-------------------------

.. automethod:: dimarray.DimArray.take

-------------------------

.. automethod:: dimarray.DimArray.put


.. _ref_api_reindexing:

Re-indexing
-----------

.. automethod:: dimarray.DimArray.reset_axis

-------------------------

.. automethod:: dimarray.DimArray.reindex_axis

-------------------------

.. automethod:: dimarray.DimArray.reindex_like

-------------------------

.. automethod:: dimarray.DimArray.sort_axis

.. _ref_api_missingvalues:

Missing values
--------------

.. automethod:: dimarray.DimArray.dropna
.. automethod:: dimarray.DimArray.fillna
.. automethod:: dimarray.DimArray.setna


To / From other objects
-----------------------

.. automethod:: dimarray.DimArray.from_pandas

-------------------------

.. automethod:: dimarray.DimArray.to_pandas

-------------------------

.. automethod:: dimarray.DimArray.to_larry

-------------------------

.. automethod:: dimarray.DimArray.to_dataset

I/O
---

.. automethod:: dimarray.DimArray.write_nc

-------------------------

.. automethod:: dimarray.DimArray.read_nc

Plotting
--------

.. automethod:: dimarray.DimArray.plot

-------------------------

.. automethod:: dimarray.DimArray.pcolor

-------------------------

.. automethod:: dimarray.DimArray.contourf

-------------------------

.. automethod:: dimarray.DimArray.contour


Comparisons
-----------

.. automethod:: dimarray.DimArray._cmp

-------------------------

.. automethod:: dimarray.DimArray.__eq__

-------------------------

.. automethod:: dimarray.DimArray.__lt__

-------------------------

.. automethod:: dimarray.DimArray.__gt__

-------------------------

.. automethod:: dimarray.DimArray.__le__

-------------------------

.. automethod:: dimarray.DimArray.__or__

-------------------------

.. automethod:: dimarray.DimArray.__and__


Unary operations
----------------

.. automethod:: dimarray.DimArray.apply

-------------------------

.. automethod:: dimarray.DimArray.__neg__

-------------------------

.. automethod:: dimarray.DimArray.__pos__

-------------------------

.. automethod:: dimarray.DimArray.__sqrt__

-------------------------

.. automethod:: dimarray.DimArray.__invert__

-------------------------

.. automethod:: dimarray.DimArray.__nonzero__


Binary operation
----------------

.. automethod:: dimarray.DimArray.__add__

-------------------------

.. automethod:: dimarray.DimArray.__sub__

-------------------------

.. automethod:: dimarray.DimArray.__mul__

-------------------------

.. automethod:: dimarray.DimArray.__div__

-------------------------

.. automethod:: dimarray.DimArray.__truediv__

-------------------------

.. automethod:: dimarray.DimArray.__floordiv__

-------------------------

.. automethod:: dimarray.DimArray.__radd__

-------------------------

.. automethod:: dimarray.DimArray.__rsub__

-------------------------

.. automethod:: dimarray.DimArray.__rmul__

-------------------------

.. automethod:: dimarray.DimArray.__rdiv__

-------------------------

.. automethod:: dimarray.DimArray.__pow__

-------------------------

.. automethod:: dimarray.DimArray.__rpow__


.. .. toctree::
..    :maxdepth: 2
.. 
.. .. automodule:: dimarray
..     :members: read_nc, stack, concatenate, broadcast_arrays, from_pandas
..     :undoc-members:
.. 
.. .. autoclass:: dimarray.DimArray
..     :members: reindex_axis, reset_axis, write_nc, mean, diff, apply, broadcast, reindex_like , reindex_axis, reset_axis, reshape, group, ungroup, swapaxes, transpose, squeeze,  to_pandas, from_pandas, to_dataset, write_nc, plot, pcolor, contourf, contour 
..     :undoc-members:
.. ..   :members: reindex_axis, reset_axis, write_nc, mean, median, max, sum, diff,  broadcast, reindex_like , reindex_axis, reset_axis, reshape, group, ungroup, swapaxes, transpose, squeeze,  to_pandas, from_pandas, to_dataset, write_nc, plot, pcolor, contourf, contour 
.. 
.. .. autoclass:: dimarray.Dataset
..    :members: to_array, write_nc, reset_axis
..    :undoc-members:

..
.. .. autoclass:: dimarray.Axis
..    :members:
..    :undoc-members:

..
.. .. autoclass:: dimarray.Axes
..    :members:
..    :undoc-members:
