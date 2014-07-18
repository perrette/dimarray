.. _ref_api_geo:

================
dimarray.geo API
================

The geo module contains a new subclass of DimArray, GeoArray, as a well
as a few specific functions.

.. contents:: 
    :local:
    :depth: 2

Create a GeoArray
-----------------

.. automethod:: dimarray.geo.GeoArray.__init__


Coordinate Reference Systems
----------------------------

.. automodule:: dimarray.geo.crs
    :members:

Projections and CRS transforms
------------------------------

.. autofunction:: dimarray.geo.transform


.. autofunction:: dimarray.geo.transform_vectors
