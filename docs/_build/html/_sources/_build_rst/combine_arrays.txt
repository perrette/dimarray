.. This file was generated automatically from the ipython notebook:
.. notebooks/combine_arrays.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

..  _page_combine_arrays:


..  _Stack_and_concatenate_arrays:

Stack and concatenate arrays
----------------------------

..  _concatenate_arrays_along_existing_axis:

concatenate arrays along existing axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import dimarray as da
>>> a = da.DimArray([[1.,2,3]],axes=[('line',[1]), ('col',['a','b','c'])])
>>> b = da.DimArray([[4,5,6],[7,8,9]], axes=[('line',[2,3]), ('col',['a','b','c'])])
>>> da.concatenate((a,b), axis=0)
dimarray: 9 non-null elements (0 null)
dimensions: 'line', 'col'
0 / line (3): 1 to 3
1 / col (3): a to c
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.],
       [ 7.,  8.,  9.]])

..  _stack_arrays_along_new_axis:

stack arrays along new axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> a = da.DimArray([10,20,30])
>>> dict_form = da.stack({'a':a, '2*a':2*a}, axis='items')   # dictionary
>>> list_form = da.stack([a, 2*a], keys=['a','2*a'], axis='items')  # list
>>> list_form
dimarray: 6 non-null elements (0 null)
dimensions: 'items', 'x0'
0 / items (2): a to 2*a
1 / x0 (3): 0 to 2
array([[10, 20, 30],
       [20, 40, 60]])

with axis alignment

>>> a = da.DimArray([10,20,30], ('x0',[0, 1, 2]))
>>> b = da.DimArray([1,2,3], ('x0', [1,2,3]))
>>> da.stack([a,b], keys=['a','b'], align=True) 
dimarray: 6 non-null elements (2 null)
dimensions: 'unnamed', 'x0'
0 / unnamed (2): a to b
1 / x0 (4): 0 to 3
array([[ 10.,  20.,  30.,  nan],
       [ nan,   1.,   2.,   3.]])