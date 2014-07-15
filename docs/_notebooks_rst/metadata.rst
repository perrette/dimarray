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

>>> a._metadata() # doctest: +SKIP
{'name': 'distance', 'units': 'meters'}

Metadata are conserved by slicing and along-axis transformation, but are lost with more ambiguous operations.

>>> a[:]._metadata()  # doctest: +SKIP
{'name': 'distance', 'units': 'meters'}

.. warning :: until version 0.1.8 metadata used to be accessed as a property (no explicit call). In subsequent versions this has been changed to a method for clarity.

A :meth:`summary` method is also defined that provide an overview of both the data and its metadata.

>>> a.axes[0].units = 'axis units'
>>> print a.summary()
dimarray: 3 non-null elements (0 null)
0 / x0 (3): 0 to 2
        units : axis units
metadata:
        units : meters
        name : distance
array([1, 2, 3])


.. _Private_attributes:

Private attributes
------------------

Metadata are nothing more than python attributes. The :meth:`_metadata` method just scans the DimArray's `__dict__` and filters out "private" attributes (start with an underscore `_`), as well as a number of special names listed in `__metadata_exclude__` class attribute. 

.. note:: Metadata that start with an underscore `_` or use any protected class attribute as name (e.g. `values`, `axes`, `dims` and so on) can be set and accessed using :meth:`set_metadata`, :meth:`get_metadata` and  :meth:`del_metadata` methods. 

>>> a.set_metadata('dims','this is a bad name')


>>> a._metadata() # doctest: +SKIP
{'dims': 'this is a bad name', 'name': 'distance', 'units': 'meters'}

>>> a.dims
('x0',)

>>> a.get_metadata('dims')
'this is a bad name'

>>> a._metadata() # doctest: +SKIP
{'dims': 'this is a bad name', 'name': 'distance', 'units': 'meters'}

Internatlly, these special metadata will be stored in a :attr:`_metadata_private` attribute (an actual dictionary, not like :meth:`_metadata`):

>>> a._metadata_private
{'dims': 'this is a bad name'}

>>> a.del_metadata('dims')


Note that :meth:`set_metadata` will first try to use python's :func:`setattr` function:.

>>> a.set_metadata('long_name','this is not a private attribute')
>>> a.set_metadata('long name','but that one is')
>>> a._metadata()
{'long name': 'but that one is',
 'long_name': 'this is not a private attribute',
 'name': 'distance',
 'units': 'meters'}

>>> a._metadata_private
{'long name': 'but that one is'}

.. _Under_the_hood:

Under the hood
--------------

:class:`DimArray`, :class:`Axis` and :class:`Dataset` all inherit from a :class:`dimarray.core.metadata.MetadataBase` which provides metadata-specific methods.