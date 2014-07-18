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

Transforms between coordinate systems
-------------------------------------

.. autofunction:: dimarray.geo.transform

.. autofunction:: dimarray.geo.transform_vectors

Coordinate Reference Systems
----------------------------

.. autoclass:: dimarray.geo.crs.Globe

.. autoclass:: dimarray.geo.crs.LatitudeLongitude
 
.. .. autoclass:: dimarray.geo.crs.Mercator

.. autoclass:: dimarray.geo.crs.PolarStereographic

.. autoclass:: dimarray.geo.crs.RotatedPole

.. autoclass:: dimarray.geo.crs.Stereographic

.. autoclass:: dimarray.geo.crs.TransverseMercator

A class to define coordinate system from PROJ.4 parameters

.. autoclass:: dimarray.geo.crs.Proj4

A function to return a CRS from various inputs (CRS, dict, str...)

.. autofunction:: dimarray.geo.crs.get_crs

.. .. automodule:: dimarray.geo.crs
..     :members:

