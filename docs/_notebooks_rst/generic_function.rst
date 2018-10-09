.. This file was generated automatically from the ipython notebook:
.. notebooks/generic_function.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_generic_function:


Create a generic time-mean function
===================================
:download:`Download notebook </notebooks/generic_function.ipynb>` 


This function applies to any kind of input array, as long as the "time" dimension is present. 

>>> def time_mean(a, t1=None, t2=None):
...     """ compute time mean between two instants
...     
...     Parameters:
...     -----------
...     a : DimArray
...         must include a "time" dimension
...     t1, t2 : same type as a.time (typically int or float)
...         start and end times
...     
...     Returns:
...     --------
...     ma : DimArray
...         time-average between t1 and t2
...     """
...     assert 'time' in a.dims, 'dimarray must have the "time" dimension'
...     return a.swapaxes(0, 'time')[t1:t2].mean(axis='time')


>>> from dimarray import DimArray
>>> import numpy as np


>>> a = DimArray([1,2,3,4], axes=[[2000,2001,2002,2003]], dims=['time'])
>>> time_mean(a, 2001, 2003)  # average over 2001, 2002, 2003
3.0

>>> a = DimArray([[1,2,3,4],[5,6,7,8]], axes=[['a','b'],[2000,2001,2002,2003]], dims=['items','time'])
>>> time_mean(a)  # average over the full time axis
dimarray: 2 non-null elements (0 null)
0 / items (2): 'a' to 'b'
array([2.5, 6.5])