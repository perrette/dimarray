==============================
DimArray methods reference API
==============================

DimArray methods are list below by topic, along with examples. 
Functions are provided in a separate page :ref:`functions`.

.. contents:: 
    :depth: 1
    :local: 

Create a DimArray
-----------------

init
~~~~
.. automethod:: dimarray.DimArray.__init__


from_kw
~~~~~~~
.. automethod:: dimarray.DimArray.from_kw

Modify shape
------------

transpose
~~~~~~~~~
.. automethod:: dimarray.DimArray.transpose

--------------------

swapaxes
~~~~~~~~
.. automethod:: dimarray.DimArray.swapaxes

--------------------

reshape
~~~~~~~
.. automethod:: dimarray.DimArray.reshape

--------------------

group
~~~~~
.. automethod:: dimarray.DimArray.group

--------------------

ungroup
~~~~~~~
.. automethod:: dimarray.DimArray.ungroup

--------------------

flatten
~~~~~~~
.. automethod:: dimarray.DimArray.flatten

--------------------

newaxis
~~~~~~~
.. automethod:: dimarray.DimArray.newaxis

--------------------

squeeze
~~~~~~~
.. automethod:: dimarray.DimArray.squeeze

--------------------

repeat
~~~~~~
.. automethod:: dimarray.DimArray.repeat

--------------------

broadcast
~~~~~~~~~
.. automethod:: dimarray.DimArray.broadcast

Reduce, accumulate
------------------

max
~~~
.. automethod:: dimarray.DimArray.max
.. 
.. --------------------
.. 
min
~~~
.. automethod:: dimarray.DimArray.min

--------------------

ptp
~~~
.. automethod:: dimarray.DimArray.ptp

--------------------

median
~~~~~~
.. automethod:: dimarray.DimArray.median

--------------------

all
~~~
.. automethod:: dimarray.DimArray.all

--------------------

any
~~~
.. automethod:: dimarray.DimArray.any

--------------------

prod
~~~~
.. automethod:: dimarray.DimArray.prod

--------------------

sum
~~~
.. automethod:: dimarray.DimArray.sum
 
--------------------

mean
~~~~
.. automethod:: dimarray.DimArray.mean

--------------------

std
~~~
.. automethod:: dimarray.DimArray.std

--------------------

var
~~~
.. automethod:: dimarray.DimArray.var

--------------------

argmax
~~~~~~
.. automethod:: dimarray.DimArray.argmax

--------------------

argmin
~~~~~~
.. automethod:: dimarray.DimArray.argmin

--------------------

cumsum
~~~~~~
.. automethod:: dimarray.DimArray.cumsum

--------------------

cumprod
~~~~~~~
.. automethod:: dimarray.DimArray.cumprod

--------------------

diff
~~~~
.. automethod:: dimarray.DimArray.diff


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
