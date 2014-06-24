.. This file was generated automatically from the ipython notebook:
.. notebooks/missingvalues.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_missingvalues:


.. _Missing_values:

Missing values
--------------

>>> import numpy as np
>>> import dimarray as da


>>> a.mean(axis=0)
dimarray: 2 non-null elements (1 null)
dimensions: 'x1'
0 / x1 (3): 0 to 2
array([ 2.5,  3.5,  nan])

>>> a.mean(axis=0, skipna=True)
dimarray: 3 non-null elements (0 null)
dimensions: 'x1'
0 / x1 (3): 0 to 2
array([ 2.5,  3.5,  3. ])

A few other methods exist but are experimental. They are mere aliases for classical `a[np.isnan[a]] = value` syntax, but automatically coerce integer type to float, and perform a copy by default. This could also be useful in the future to define a missing value flag other than `nan`, for example when working with integer array.

>>> a.fillna(99)
dimarray: 6 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (3): 0 to 2
array([[  1.,   2.,   3.],
       [  4.,   5.,  99.]])

`setna` can also be provided with a list of values (or boolean arrays) to set to nan:

>>> b = a.setna([1,4])
>>> b
dimarray: 3 non-null elements (3 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (3): 0 to 2
array([[ nan,   2.,   3.],
       [ nan,   5.,  nan]])

More interestingly, the `dropna` methods helps getting rid of nans arising in grouping operations, similarly to `pandas`:

>>> b.dropna(axis=1)
dimarray: 2 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (1): 1 to 1
array([[ 2.],
       [ 5.]])

But in some cases, you are still ok with a certain number of nans, but want to have a minimum of 1 or more valid values:

>>> b.dropna(axis=1, minvalid=1)  # minimum number of valid values, equivalent to `how="all"` in pandas
dimarray: 3 non-null elements (1 null)
dimensions: 'x0', 'x1'
0 / x0 (2): 0 to 1
1 / x1 (2): 1 to 2
array([[  2.,   3.],
       [  5.,  nan]])