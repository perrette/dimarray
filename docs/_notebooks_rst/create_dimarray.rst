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
0 / variable (2): 'a' to 'b'
1 / time (3): 1950 to 1970
array([[1., 2., 3.],
       [4., 5., 6.]])

.. _List_of_tuples:

List of tuples
--------------

DimArray axes can also be initialized via a list of tuples (axis name, axis values):

>>> a = DimArray([[1.,2,3], [4,5,6]], axes=[('variable', ['a', 'b']), ('time', [1950, 1960, 1970])])
>>> a
dimarray: 6 non-null elements (0 null)
0 / variable (2): 'a' to 'b'
1 / time (3): 1950 to 1970
array([[1., 2., 3.],
       [4., 5., 6.]])

.. _Recursive_definition___dict_of_dict:

Recursive definition : dict of dict
-----------------------------------

.. versionadded :: 0.1.8

It is possible to define a dimarray as a dictionary of dictionary. The only additional parameter needed is a list of dimension names, that should correspond to the dictionary's depth. 

>>> dict_ = {'a': {1:11,
...                2:22,
...                3:33},
...          'b': {1:111,
...                2:222,
...                3:333} }
>>> a = DimArray(dict_, dims=['dim1','dim2'])
>>> a.sort_axis(axis=0).sort_axis(axis=1)  # dict keys are not sorted in python !
dimarray: 6 non-null elements (0 null)
0 / dim1 (2): 'a' to 'b'
1 / dim2 (3): 1 to 3
array([[ 11,  22,  33],
       [111, 222, 333]])