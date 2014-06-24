.. This file was generated automatically from the ipython notebook:
.. notebooks/create_dimarray.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_create_dimarray:


Create a dimarray
=================
:download:`Download notebook </notebooks/create_dimarray.ipynb>` 


There are various ways of defining a DimArray instance. 

.. _Standard_definition:

Standard definition
-------------------

Provide a list of axis values (`axes=` parameter) and a list of axis names (`dims=`) parameter. 

>>> from dimarray import DimArray
>>> a = DimArray([[1.,2,3], [4,5,6]], axes=[['a', 'b'], [1950, 1960, 1970]], dims=['variable', 'time'])
>>> a
dimarray: 6 non-null elements (0 null)
dimensions: 'variable', 'time'
0 / variable (2): a to b
1 / time (3): 1950 to 1970
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])

.. _List_of_tuples:

List of tuples
--------------

DimArray axes can also be initialized via a list of tuples (axis name, axis values):

>>> a = DimArray([[1.,2,3], [4,5,6]], axes=[('variable', ['a', 'b']), ('time', [1950, 1960, 1970])])
>>> a
dimarray: 6 non-null elements (0 null)
dimensions: 'variable', 'time'
0 / variable (2): a to b
1 / time (3): 1950 to 1970
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])