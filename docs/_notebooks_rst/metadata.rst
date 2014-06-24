.. This file was generated automatically from the ipython notebook:
.. notebooks/metadata.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_metadata:


.. _Metadata:

Metadata
~~~~~~~~
:download:`Download notebook </notebooks/metadata.ipynb>` 


:class:`dimarray.DimArray`, :class.`dimarray.Dataset` and :class:`dimarray.Axis` all support metadata. The straightforward way to define them is via the standard `.` syntax to access an object attribute:

>>> from dimarray import DimArray
>>> a = DimArray([1,2,3])


>>> a.name = 'myname'
>>> a.units = 'myunits'


The `_metadata` property returns a dictionary of metadata, for checking:

>>> a._metadata # doctest: +SKIP
{'name': 'myname', 'units': 'myunits'}

Metadata are conserved by slicing and along-axis transformation, but are lost with any other transformation.

>>> a[:]._metadata  # doctest: +SKIP
{'name': 'myname', 'units': 'myunits'}

.. note:: Currently metadata are not stored in the `_metadata` attribute but quite classically in the class's `__dict__` attribute. `_metadata` is only a convenience property that makes a copy of all non-private instance attributes. Therefore modifying its values element-wise will have no effect on actual metadata. 

.. note:: Any attribute starting with  `_` will not show up in `_metadata`. This `private` attributes will not be conserved via indexing or transformation, and will not be written to netCDF. They can still be read from a netCDF file, though.
