.. This file was generated automatically from the ipython notebook:
.. notebooks/geoarray.ipynb
.. To modify this file, edit the source notebook and execute "make rst"

.. _page_geoarray:


The geo sub-package.
====================
:download:`Download notebook </notebooks/geoarray.ipynb>` 


.. versionadded:: 0.1.9

:class:`dimarray.geo.GeoArray` is a subclass of :class:`dimarray.DimArray` that is more specific to geoscientifical applications. The most recognizable features are an automatic check for longitude and latitude coordinates.

>>> from dimarray.geo import GeoArray


>>> a = GeoArray([0,0,0], axes=[('lon',[-180,0,180])])
>>> a
geoarray: 3 non-null elements (0 null)
0 / lon (3): -180.0 to 180.0 (Longitude)
array([0, 0, 0])

There are also pre-defined axes which key-word arguments to ease the definition of common arrays

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

.. note:: The keyword arguments assume an order `time`, `lat`, `lon` following common usage and CF-recommandations. Additional predefined coordinate axes are `z`,`y` and `x` to indicate vertical and horizontal coordinates, more generally. 

.. note :: For now there is no constraint on the coordinate axis. This might change in the future, by imposing a strict ordering relationship. 

Functions and methods under dimarray.geo are identical or modified from standard dimarray to account for 

.. _Coordinate_classes:

Coordinate classes
------------------

Under the hood, there are new :class:`Coordinate` classes which inherit from :class:`Axis`.

For example, the inheritance relations of Latitude is: :class:`Latitude` -> :class:`Y` -> :class:`Coordinate` -> :class:`Axis`.

>>> from dimarray.geo import Latitude, Y, Coordinate, Axis


>>> assert isinstance(a.axes['lat'], Latitude) 
>>> assert issubclass(Latitude, Y) 
>>> assert issubclass(Y, Coordinate) 
>>> assert issubclass(Coordinate, Axis)


The advantage of this architecture is that specific properties such as weights or 360-modulo indexing are automatically defined. 

>>> a.axes['lat'].weights  # lat -> cos(lat) weighted mean
<function dimarray.geo.geoarray.<lambda>>

>>> a.axes['lon'].modulo
360.0

In the case of Latitude and Longitude, some metadata are also provided by default.

>>> a.axes['lat']._metadata()
{'long_name': 'latitude',
 'standard_name': 'latitude',
 'units': 'degrees_north'}

.. _Projections:

Projections
-----------

Transformation between various coordinate reference systems is addressed in the chapter :ref:`projection`.