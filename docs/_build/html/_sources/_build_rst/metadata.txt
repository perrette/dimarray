.. This file was generated automatically from the ipython notebook:
.. notebooks/metadata.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

..  _page_metadata:


..  _Metadata:

Metadata
~~~~~~~~

`DimArray` and `Axis` objects, support metadata. They can be passed by keyword arguments to DimArray (not via da.array_kw or DimArray.from_kw NOTE: may remove this functionality), or afterwards:

>>> from dimarray import DimArray
>>> a = DimArray([[1,2,3],[4,5,6]])
>>> a.name='myname'
>>> a.units='myunits'


>>> ax = a.axes[0]
>>> ax.units = "meters"


metadata are conserved by slicing and along-axis transformation, but are lost with any other transformation

>>> a[:].units
'myunits'

>>> ax[:].units
'meters'