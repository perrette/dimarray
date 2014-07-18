.. _ref_api_geo:

================
dimarray.geo API
================

The geo module contains a new subclass of DimArray, GeoArray, as a well
as a few specific functions.

.. contents:: 
    :local:
    :depth: 2

GeoArray and Coordinate classes
-------------------------------

.. autoclass:: dimarray.geo.GeoArray
    :show-inheritance: 

---------------------------------------

.. autoclass:: dimarray.geo.Coordinate
    :show-inheritance: 

---------------------------------------

.. autoclass:: dimarray.geo.Time
    :show-inheritance: 

---------------------------------------

.. autoclass:: dimarray.geo.Z
    :show-inheritance: 

---------------------------------------

.. autoclass:: dimarray.geo.Y
    :show-inheritance: 

---------------------------------------

.. autoclass:: dimarray.geo.X
    :show-inheritance: 

---------------------------------------

.. autoclass:: dimarray.geo.Longitude
    :show-inheritance: 
    :undoc-members:


---------------------------------------

.. autoclass:: dimarray.geo.Latitude
    :show-inheritance: 
    :undoc-members:

Transforms between coordinate systems
-------------------------------------

.. autofunction:: dimarray.geo.transform

---------------------------------------

.. autofunction:: dimarray.geo.transform_vectors

Coordinate Reference Systems
----------------------------

.. autoclass:: dimarray.geo.crs.Globe
    :show-inheritance: 
    :no-undoc-members: from_proj4

---------------------------------------

.. autoclass:: dimarray.geo.crs.LatitudeLongitude
    :show-inheritance: 
    :members:

---------------------------------------
 
.. autoclass:: dimarray.geo.crs.PolarStereographic
    :show-inheritance: 
    :members:

---------------------------------------

.. autoclass:: dimarray.geo.crs.RotatedPole
    :show-inheritance: 
    :members:

---------------------------------------

.. autoclass:: dimarray.geo.crs.Stereographic
    :show-inheritance: 
    :members:

---------------------------------------

.. autoclass:: dimarray.geo.crs.TransverseMercator
    :show-inheritance: 
    :members:

---------------------------------------

A class to define coordinate system from PROJ.4 parameters

.. autoclass:: dimarray.geo.crs.Proj4
    :show-inheritance: 
    :members:

---------------------------------------

A function to return a CRS from various inputs (CRS, dict, str...)

.. autofunction:: dimarray.geo.crs.get_crs
