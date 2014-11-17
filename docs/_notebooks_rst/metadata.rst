.. This file was generated automatically from the ipython notebook:
.. notebooks/metadata.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_metadata:


Metadata
========
:download:`Download notebook </notebooks/metadata.ipynb>` 


:class:`DimArray`, :class:`Dataset` and :class:`Axis` all support metadata. The straightforward way to define them is via the standard `.` syntax to access an object attribute:

>>> from dimarray import DimArray
>>> a = DimArray([1,2,3])


>>> a.name = 'distance'
>>> a.units = 'meters'


Although they are nothing more than usual python attributes, the :meth:`_metadata` method gives an overview of all metadata:

>>> a.attrs
OrderedDict([('name', 'distance'), ('units', 'meters')])

Metadata are conserved by slicing and along-axis transformation, but are lost with more ambiguous operations.

>>> a[:].attrs
OrderedDict([('name', 'distance'), ('units', 'meters')])

>>> (a**2).attrs 
OrderedDict()

.. warning :: The `attrs` attribute has been added in version 0.2, thereby deprecating the former _metadata.

A :meth:`summary` method is also defined that provide an overview of both the data and its metadata.

>>> a.axes[0].units = 'axis units'
>>> a.summary()
dimarray: 3 non-null elements (0 null)
0 / x0 (3): 0 to 2
    units: 'axis units'
attributes:
    name: 'distance'
    units: 'meters'
array([1, 2, 3])


.. note:: Metadata that start with an underscore `_` or use any protected class attribute as name (e.g. `values`, `axes`, `dims` and so on) can be set and accessed using by manipulating :attr:`attrs`.

>>> a.attrs['dims'] = 'this is a bad name'


>>> a.attrs 
OrderedDict([('name', 'distance'), ('units', 'meters'), ('dims', 'this is a bad name')])

>>> a.dims
('x0',)

It is easy to clear metadata:

>>> a.attrs = {} # clean all metadata
