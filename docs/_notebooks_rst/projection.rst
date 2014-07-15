.. This file was generated automatically from the ipython notebook:
.. notebooks/projection.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_projection:


.. _projection:

Coordinate reference systems and Projections
============================================
:download:`Download notebook </notebooks/projection.ipynb>` 


dimarray.geo is shipped with :func:`dimarray.geo.transform` and :func:`dimarray.geo.transform_vectors` functions to handle transformations across coordinate reference systems. It is based on `cartopy.crs` methods, itself built on `PROJ.4` library.

In contrast to cartopy/PROJ.4, dimarray.geo functions perform both coordinate transforms and regridding onto a regular grid in the new coordinate system. This is because of the structure of DimArray and GeoArray classes, which only accept regular coordinates.

>>> from dimarray.geo 
