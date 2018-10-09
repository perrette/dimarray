.. include:: ../README.rst

Packages you might also be interested in
----------------------------------------

`dimarray`_ is built on top of `numpy`_, as an alternative to `larry`_ and `pandas`_

dimarray's default indexing method is on labels, which makes it very useful as 
data structure to store high dimensional problems with few labels, such 
as sensitivity analysises (or e.g. climate scenarios...).

If your focus is on large geoscientific data however, `xarray`_ 
is a more appropriate package, with useful methods to load large datasets, 
and a datamodel closely aligned on the netCDF.
Moreover, standard, numpy-like integer indexing is more apppropriate for geographic maps.

`pandas`_ is an excellent package for tabular data analysis, 
supporting many I/O formats and axis alignment (or "reindexing") 
in binary operations. It is mostly limited to 2 dimensions (DataFrame), 
or up to 4 dimensions (Panel, Panel4D).  

.. _numpy: http://docs.scipy.org/doc/numpy/user
.. _larry: http://berkeleyanalytics.com/la
.. _pandas: http://pandas.pydata.org 
.. _dimarray: dimarray.readthedocs.org
.. _xarray: xarray.readthedocs.org

.. seealso:: :meth:`DimArray.to_pandas() <dimarray.DimArray.to_pandas>` and :meth:`DimArray.from_pandas() <dimarray.DimArray.from_pandas>`.
