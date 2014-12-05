""" Subclass of DimArray specialized for geo-applications

==> dimensions always remain sorted: ('items','time','lat','lon','height','sample')
==> subset of methods with specific plotting behaviour: e.g. TimeSeries, Map

References
----------
    http://cfconventions.org/1.4
    http://cfconventions.org/1.6 (in track-change mode w.r.t CF-1.4)
"""
from __future__ import absolute_import
from collections import OrderedDict as odict
import sys
import numpy as np
from dimarray import DimArray, Axes, Axis

def is_longitude(nm):
   return nm.lower() in ("lon","long", "longitude", "lons", "longitudes")

def is_latitude(nm):
   return nm.lower() in ("lat", "latitude", "lats", "latitudes")

def is_time(nm):
   return nm.lower() in ("time",) # experimental, may be extended

class GeoArray(DimArray):
    """ Subclass of DimArray specialized for geo-scientific use
    
    Its main features are :

    - recognizes space-time coordinates (e.g. lon, lat)
    - add convenience key-word arguments for coordinate axes and metadata
    """
    __metadata_include__ = ['grid_mapping', 'long_name', 'units', 'standard_name']

    # units = None
    # long_name = None
    # standard_name = None

    grid_mapping = None

    def __init__(self, values=None, axes=None, 
            time=None, z=None, y=None, x=None, lat=None, lon=None, 
            dims=None, 
            standard_name=None, long_name=None, units=None,
            **kwargs):
        """ 
        Parameters
        ----------
        values : array-like
        axes : list of array-like or Axis objects, optional
        time, z, y, x, lat, lon: spatiotemporal coordinates, optional
           These keyword arguments are provided for convenience. They
           can be used instead of (but not in addition to) the `axes` 
           parameter. See notes below about implied array shape.
        dims : sequence, optional
            sequence of axis names (dimensions)
            This is needed when axes are defined via keywords but 
            array-shape does not conform with CF-conventions.
        standard_name, long_name, units : str, optional
            standard attributes according to the CF-1.4 netCDF 
            conventions. Will be added to metadata.
        **kwargs : keyword arguments, optional
            Passed to dimarray.DimArray

        Notes
        -----
        When coordinate axes are passed via keyword arguments 
        it is assumed that the array shape follows the CF-conventions:
        time, vertical coordinate (z), northing coordinate (y or lat), 
        easting coordinate (x or lon).
        If it is not the case, please indicate the `dims` parameters or 
        pass axes via the `axes=` parameter instead of keyword arguments.

        See Also
        --------
        dimarray.DimArray : base class with no "geo" specificities
        dimarray.geo.Coordinate : base class for geo-axes
        dimarray.geo.Time 
        dimarray.geo.Latitude, dimarray.geo.Longitude
        dimarray.geo.Z, dimarray.geo.Y, dimarray.geo.X
        """
        keyword = (time is not None or lat is not None or lon is not None 
                or x is not None or y is not None or z is not None)
        if axes is not None and keyword:
            msg = "Axes can be provided EITHER via `axes=` OR keyword arguments"
            raise ValueError(msg)

        # construct the axes
        if keyword:
            axes = Axes()
            if time is not None: axes.append(Time(time, 'time'))
            if z is not None: axes.append(Z(z, 'z'))
            if y is not None: axes.append(Y(y, 'y'))
            if lat is not None: axes.append(Latitude(lat, 'lat'))
            if x is not None: axes.append(X(x, 'x'))
            if lon is not None: axes.append(Longitude(lon, 'lon'))
            if dims is not None:
                if len(dims) != len(axes):
                    msg = "dims ({}) and axes ({}) lengths \
                            do not match".format(len(dims), len(axes))
                    raise ValueError(msg)
                axes = [axes[nm] for nm in dims]

        #if metadata is not None:
        #    for k in metadata.keys():
        #        assert k not in kwargs, "{} was provided multiple times".format(k)
        #    kwargs.update(metadata) # TODO: make metadata a parameter in DimArray as well

        self._grid_mapping = None

        super(GeoArray, self).__init__(values, axes, **kwargs) # order dimensions

        # add metadata
        if units: self.units = units
        if standard_name: self.standard_name = standard_name
        if long_name: self.long_name = long_name

        # Do some guessing to define coordinates
        for i, ax in enumerate(self.axes):
            if isinstance(ax, Coordinate):
                continue
            elif is_latitude(ax.name) or (hasattr(ax, 'standard_name') \
                    and ax.standard_name == 'latitude'):
                self.axes[i] = Latitude.from_axis(ax)
            elif is_longitude(ax.name) or (hasattr(ax, 'standard_name') \
                    and ax.standard_name == 'longitude'):
                self.axes[i] = Longitude.from_axis(ax)
            elif is_time(ax.name):
                self.axes[i] = Time.from_axis(ax)
            # 'x', 'y', 'z' are too general to be used.
            elif ax.name == 'x' or (hasattr(ax, 'standard_name') \
                    and ax.standard_name == 'projection_x_coordinate'):
                self.axes[i] = X.from_axis(ax)
            elif ax.name == 'y' or (hasattr(ax, 'standard_name') \
                    and ax.standard_name == 'projection_y_coordinate'):
                self.axes[i] = Y.from_axis(ax)
            elif ax.name in ('z','height','depth'):
                self.axes[i] = Z.from_axis(ax)


        # Check unicity of coordinates.
        time_coords = filter(lambda ax : isinstance(ax, Time), self.axes)
        x_coords = filter(lambda ax : isinstance(ax, X), self.axes)
        y_coords = filter(lambda ax : isinstance(ax, Y), self.axes)
        z_coords = filter(lambda ax : isinstance(ax, Z), self.axes)

        if len(time_coords) > 1:
            raise ValueError("More than one time coordinate found")
        if len(x_coords) > 1:
            raise ValueError("More than one x coordinate found")
        if len(y_coords) > 1:
            raise ValueError("More than one y coordinate found")
        if len(z_coords) > 1:
            raise ValueError("More than one z coordinate found")

    def __repr__(self): return super(GeoArray, self).__repr__().replace("dimarray","geoarray")
    def __print__(self): return super(GeoArray, self).__print__().replace("dimarray","geoarray")
    def __str__(self): return super(GeoArray, self).__str__().replace("dimarray","geoarray")

    @property
    def grid_mapping(self):
        if 'grid_mapping' in self.attrs:
            return self.attrs['grid_mapping'] 
        else:
            return self._grid_mapping

    @grid_mapping.setter
    def grid_mapping(self, val):
        if isinstance(val, basestring): # e.g. variable name in a dataset
            self.attrs['grid_mapping'] = val
        else:
            try: 
                from .crs import get_crs
                self._grid_mapping = get_crs(val)
                self.attrs.pop('grid_mapping', None) # remove from attrs
            except:
                self.attrs['grid_mapping'] = val
                warnings.warn("Problem when setting the grid mapping")


# define Latitude and Longitude axes
class Coordinate(Axis):
    """ Subclass of Axis representing a space-time coordinate.
    """
    # _attrs = None
    _name = None
    _standard_name = None
    _long_name = None
    _units = None

    __metadata_include__  = ['standard_name', 'long_name', 'units'] # may be defined as class attributes

    def __init__(self, values, name=None, dtype=float, **kwargs):

        # Initialize the metadata with non-None class values
        metadata = {k:getattr(self, '_'+k) for k in self.__metadata_include__ if getattr(self, '_'+k) is not None}

        metadata.update(kwargs)

        if name is None:
            name = self._name # class attribute

        super(Coordinate, self).__init__(values, name, dtype=dtype, **metadata)

    @classmethod
    def from_axis(cls, ax):
        " define a Coordinate from an Axis object "
        return cls(ax.values, ax.name, **ax.attrs)

    def _repr(self, metadata=None):
        return super(Coordinate, self)._repr(metadata)+" ({})".format(self.__class__.__name__)


class Time(Coordinate):
    """ Time coordinate
    """
    _name = "time"

#
# Spatial coordinates
# 
class X(Coordinate):
    """ X-horizontal coordinate on the projection plane
    """
    _name = 'x'
    _long_name = "horizontal x coordinate"

class Y(Coordinate):
    """ Y-horizontal coordinate on the projection plane
    """
    _name = 'y'
    _long_name = "horizontal y coordinate"

class Z(Coordinate):
    """ Vertical coordinate
    """
    _name = 'z'
    _long_name = "vertical z coordinate"

# Longitude and Latitude as subclasses

class Latitude(Y):
    """ Latitude Coordinate
    """
    _name = 'lat'
    _long_name = "latitude"
    _units = "degrees_north"
    _standard_name = "latitude"

    def __init__(self, *args, **kwargs):
        # TODO: may need to perform additional test on sorted axes and so on
        # weighted mean with cos(lat) does not make sense for a sample of latitude
        if not 'weights' in kwargs:
            kwargs['weights'] = lambda x: np.cos(np.radians(x))
        return super(Latitude, self).__init__(*args, **kwargs)
         
class Longitude(X):
    """ Longitude Coordinate
    """
    _name='lon' 
    _long_name="longitude" 
    _units="degrees_east"
    _standard_name="longitude" 
