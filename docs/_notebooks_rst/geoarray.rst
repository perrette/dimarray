.. This file was generated automatically from the ipython notebook:
.. notebooks/geoarray.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_geoarray:


The geo sub-package
===================
:download:`Download notebook </notebooks/geoarray.ipynb>` 


.. versionadded:: 0.1.9

.. contents::
    :local:

:class:`dimarray.geo.GeoArray` is a subclass of :class:`dimarray.DimArray` that is more specific to geoscientific applications. The most recognizable features are automatic checks for longitude and latitude coordinates.

>>> from dimarray.geo import GeoArray


>>> a = GeoArray([0,0,0], axes=[('lon',[-180,0,180])])
>>> a
geoarray: 3 non-null elements (0 null)
0 / lon (3): -180.0 to 180.0 (Longitude)
array([0, 0, 0])

Coordinate axes can now be defined as keyword arguments:

>>> import numpy as np


>>> a = GeoArray(np.ones((2,3,4)), time=[1950., 1960.], lat=np.linspace(-90,90,3), lon=np.linspace(-180,180,4))
>>> a
geoarray: 24 non-null elements (0 null)
0 / time (2): 1950.0 to 1960.0 (Time)
1 / lat (3): -90.0 to 90.0 (Latitude)
2 / lon (4): -180.0 to 180.0 (Longitude)
array([[[ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.]],
<BLANKLINE>
       [[ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.]]])

.. note:: The keyword arguments assume an order time (`time`), vertical dimension (`z`), horizontal northing dimension (`lat` or `y`) and  horizontal easting dimension (`x` or `lon`), following CF-recommandations. 

All standard dimarray functions are available under dimarray.geo (so that `import dimarray.geo as da` works), and a few functions or classes such as :func:`read_nc` or :class:`Dataset` are modified to return :class:`GeoArray` instead of :class:`DimArray` instances.

.. _Coordinate_Axes:

Coordinate Axes
---------------

Under the hood, there are new :class:`Coordinate` classes which inherit from :class:`Axis`.

For example, the inheritance relations of Latitude is: :class:`Latitude` -> :class:`Y` -> :class:`Coordinate` -> :class:`Axis`.

>>> from dimarray.geo import Latitude, Y, Coordinate, Axis


>>> assert isinstance(a.axes['lat'], Latitude) 
>>> assert issubclass(Latitude, Y) 
>>> assert issubclass(Y, Coordinate) 
>>> assert issubclass(Coordinate, Axis)


The advantage of this architecture is that specific properties such as weights or 360-modulo indexing are automatically defined. 

>>> a.axes['lat'].weights  # lat -> cos(lat) weighted mean # doctest: +SKIP
<function dimarray.geo.geoarray.<lambda>>

>>> a.axes['lon'].modulo
360.0

In the case of Latitude and Longitude, some metadata are also provided by default.

>>> a.axes['lat']._metadata()  # doctest: +SKIP
{'long_name': 'latitude',
 'standard_name': 'latitude',
 'units': 'degrees_north'}

.. note :: For now there is no constraint on the coordinate axis. This might change in the future, by imposing a strict ordering relationship. 

.. seealso:: :ref:`ref_api_geo`

.. _Projections:

Projections
-----------

dimarray.geo is shipped with :func:`dimarray.geo.transform` and :func:`dimarray.geo.transform_vectors` functions to handle transformations across coordinate reference systems. They are based on :class:`cartopy.crs.CRS`. Cartopy itself makes use of the `PROJ.4` library. In addition to the list of cartopy projections, the :class:`dimarray.geo.crs.Proj4` class makes it possible to define a projection directly from `PROJ.4 parameters <https://trac.osgeo.org/proj/wiki/GenParms>`_. For the most common projections, :mod:`dimarray.geo.crs` also provides wrapper classes that can be initialized with `CF parameters <http://cfconventions.org>`_. See :func:`dimarray.geo.crs.get_crs` for more information.

In contrast to cartopy/PROJ.4, dimarray.geo functions perform both coordinate transforms and regridding onto a regular grid in the new coordinate system. This is because of the structure of DimArray and GeoArray classes, which only accept regular grids (in the sense of a collection of 1-D axes).

.. note :: Why cartopy and not just pyproj? Pyproj would be just fine, and is more minimalistic, but cartopy also implements vector transformas and offers other useful features related to plotting, reading shapefiles, download online data and so on, which come in handy. Moreover it feels more `"pythonic" <http://legacy.python.org/dev/peps/pep-0008>`_, is actively developed with support from the Met' Office, and is related to another interesting project, iris. It builds on other powerful packages such as shapely and it feels like in the long (or not so long) run it might grow toward something even more useful.

.. seealso:: :ref:`projection`