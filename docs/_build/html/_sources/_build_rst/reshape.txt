.. This file was generated automatically from the ipython notebook:
.. notebooks/reshape.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

..  _page_reshape:


..  _Modify_array_shape:

Modify array shape
------------------

Basic numpy methods to modify array dimensions are implemented in dimarray, with some additional functionality allowed by named dimensions.

**Methods overview**: :py:meth:`dimarray.DimArray.newaxis`, :py:meth:`dimarray.DimArray.squeeze`, :py:meth:`dimarray.DimArray.transpose`, :py:meth:`dimarray.DimArray.swapaxes`, :py:meth:`dimarray.DimArray.flatten`, :py:meth:`dimarray.DimArray.group`, :py:meth:`dimarray.DimArray.ungroup`, :py:meth:`dimarray.DimArray.reshape`

..  _transpose:

transpose
~~~~~~~~~

Transpose, just like its numpy equivalent, permutes dimensions, but in dimarray it can be provided with axis names instead of just axis position.

>>> from dimarray import DimArray
>>> a = DimArray([[1,2,3],[3,4,5]],dims=('x0','x1'))
>>> a.transpose()
>>> a.T
dimarray: 6 non-null elements (0 null)
dimensions: 'x1', 'x0'
0 / x1 (3): 0 to 2
1 / x0 (2): 0 to 1
array([[1, 3],
       [2, 4],
       [3, 5]])

>>> a = DimArray([[[1,2,3],[3,4,5]]],dims=('x2','x0','x1'))
>>> a.transpose('x1','x2','x0')
dimarray: 6 non-null elements (0 null)
dimensions: 'x1', 'x2', 'x0'
0 / x1 (3): 0 to 2
1 / x2 (1): 0 to 0
2 / x0 (2): 0 to 1
array([[[1, 3]],
<BLANKLINE>
       [[2, 4]],
<BLANKLINE>
       [[3, 5]]])

..  _swapaxes:

swapaxes
~~~~~~~~

Sometimes it is only useful to have on dimension in the first position, for example to make indexing easier. 
`swapaxes` is a more general method of swapping two axes, but it can achieve that operation nicely (more useful with more than 2 dimensions!):

>>> a = DimArray([[1,2,3],[3,4,5]],dims=('x0','x1'))
>>> a.swapaxes('x1',0)
dimarray: 6 non-null elements (0 null)
dimensions: 'x1', 'x0'
0 / x1 (3): 0 to 2
1 / x0 (2): 0 to 1
array([[1, 3],
       [2, 4],
       [3, 5]])

..  _group_and_ungroup_[experimental]:

group and ungroup [experimental]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a new, experimental feature, it is possible to flatten (group) or any subset of dimensions. Corresponding axes are converted in GroupedAxis objects. 

>>> import numpy as np
>>> data = np.arange(2*3*4).reshape(2,3,4)
>>> v = DimArray(data, dims=['time','lat','lon'], axes=[[1950,1955], np.linspace(-90,90,3), np.linspace(-180,180,4)])
>>> v
dimarray: 24 non-null elements (0 null)
dimensions: 'time', 'lat', 'lon'
0 / time (2): 1950 to 1955
1 / lat (3): -90.0 to 90.0
2 / lon (4): -180.0 to 180.0
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
<BLANKLINE>
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])

Flatten a set of dimensions:

>>> w = v.group(('lat','lon'))
>>> w
dimarray: 24 non-null elements (0 null)
dimensions: 'time', 'lat,lon'
0 / time (2): 1950 to 1955
1 / lat,lon (12): (-90.0, -180.0) to (90.0, 180.0)
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])

Along-axis transformations use that feature and can group any subset of axes prior to the operation:

>>> v.mean(axis=('lat','lon'))
dimarray: 2 non-null elements (0 null)
dimensions: 'time'
0 / time (2): 1950 to 1955
array([  5.5,  17.5])

Any grouped axis can be reshaped back to full n-d array via **`ungroup`**

>>> w.ungroup()
dimarray: 24 non-null elements (0 null)
dimensions: 'time', 'lat', 'lon'
0 / time (2): 1950 to 1955
1 / lat (3): -90.0 to 90.0
2 / lon (4): -180.0 to 180.0
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
<BLANKLINE>
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])

..  _reshape_[experimental]:

reshape [experimental]
~~~~~~~~~~~~~~~~~~~~~~

`reshape` is similar but not the same as numpy ndarray's `reshape`. It takes only axis names as parameters. It is a high-level function that combine `newaxis`, `squeeze`, `group` and `ungroup` to reshape the array. It differs from numpy in that it cannot "break" an existing dimension (unless it is a GroupedAxis). Provided with the parameter `transpose=True`, it also performs transpose as needed to match the required shape. 

Here an example where high-dimensional data is converted into a pandas' DataFrame for displaying result of a sensitivity analysis. GroupedAxis are converted into MultiIndex before passing to pandas.

>>> large_array = DimArray(np.arange(2*2*5*2).reshape(2,2,5,2), dims=('A','B','C','D'))
>>> large_array.reshape('A,B','C,D').to_pandas()
C     0       1       2       3       4    
D     0   1   0   1   0   1   0   1   0   1
A B                                        
0 0   0   1   2   3   4   5   6   7   8   9
  1  10  11  12  13  14  15  16  17  18  19
1 0  20  21  22  23  24  25  26  27  28  29
  1  30  31  32  33  34  35  36  37  38  39


