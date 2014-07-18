""" Subclass of DimArray specialized for geo-applications

==> dimensions always remain sorted: ('items','time','lat','lon','height','sample')
==> subset of methods with specific plotting behaviour: e.g. TimeSeries, Map

References
----------
    http://cfconventions.org/1.4
    http://cfconventions.org/1.6 (in track-change mode w.r.t CF-1.4)
"""
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
    """ array for geophysical application
    
    recognize longitude / latitude:
    - automatically assign a weight to lon varying with cos of lat, so that mean() 
    returns a good approximation to a calculation in spherical coordinates.
    - lon recognized as 360-modulo-axis: indexing -180 is the same as +180
    - can input time, lat, lon as keyword arguments in addition to the standard way
    """
    __metadata_include__ = ['grid_mapping', 'long_name', 'units', 'standard_name']

    units = None
    long_name = None
    standard_name = None

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

        Note
        ----
        Note that when coordinate axes are passed via keyword arguments 
        it is assumed that the array shape follows the CF-conventions:
        time, vertical coordinate (z), northing coordinate (y or lat), 
        easting coordinate (x or lon).
        If it is not the case, please indicate the `dims` parameters or provide

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


# define Latitude and Longitude axes
class Coordinate(Axis):
    """ Axis with a geophysical meaning, following the netCDF CF conventions
    """
    standard_name = None
    long_name = None
    units = None

    __metadata_include__  = ['standard_name', 'long_name', 'units'] # may be defined as class attributes

    def __init__(self, values, name=None, dtype=float, standard_name="", long_name="", units="", **kwargs):

        # Initialize the metadata with non-None class values
        metadata = {k:getattr(self, k) for k in self.__metadata_include__ if getattr(self, k) is not None}

        if name is None:
            name = self.name # class attribute

        if long_name: 
            metadata['long_name'] = long_name

        if units: # user-provided
            metadata['units'] = units

        if standard_name: 
            metadata['standard_name'] = standard_name

        # NOTE: This may change in the future as metadata may be provided 
        # as separate `metadata` parameter
        # There would be a need then to separate parameters (e.g. weights)
        # from metadata (e.g. units)
        metadata.update(kwargs)

        #cls = self.__class__
        #module = sys.modules[__name__]
        #cls = getattr(module, self.__class__.__name__) # ?
        #print name
        super(Coordinate, self).__init__(values, name, dtype=dtype, **metadata)
        #Axis.__init__(self, values, name, dtype=dtype, **metadata) 

    @classmethod
    def from_axis(cls, ax):
        " define a Coordinate from an Axis object "
        return cls(ax.values, ax.name, **ax._metadata)

    def _repr(self, metadata=None):
        return super(Coordinate, self)._repr(metadata)+" ({})".format(self.__class__.__name__)


class Time(Coordinate):
    """ Time coordinate

    Reference
    ---------
    http://cfconventions.org/1.6#time-coordinate
    """
    name = "time"

#
# Spatial coordinates
# 
class X(Coordinate):
    """ X-horizontal coordinate on the projection plane

    Reference
    ---------
    http://cfconventions.org/1.6#grid-mappings-and-projections
    """
    name = 'x'
    long_name = "horizontal x coordinate"

class Y(Coordinate):
    """ Y-horizontal coordinate on the projection plane

    Reference
    ---------
    http://cfconventions.org/1.6#grid-mappings-and-projections
    """
    name = 'y'
    long_name = "horizontal y coordinate"

class Z(Coordinate):
    """ Vertical coordinate

    Reference
    ---------
    http://cfconventions.org/1.6#vertical-coordinate "
    """
    name = 'z'
    long_name = "vertical z coordinate"

# Longitude and Latitude as subclasses

class Latitude(Y):
    """ Latitude Coordinate

    Reference
    ---------
    http://cfconventions.org/1.6#latitude-coordinate
    """
    name = 'lat'
    long_name = "latitude"
    units = "degrees_north"
    standard_name = "latitude"

    def __init__(self, *args, **kwargs):
        # TODO: may need to perform additional test on sorted axes and so on
        # weighted mean with cos(lat) does not make sense for a sample of latitude
        if not 'weights' in kwargs:
            kwargs['weights'] = lambda x: np.cos(np.radians(x))
        return super(Latitude, self).__init__(*args, **kwargs)
         
class Longitude(X):
    """ Longitude Coordinate

    Reference
    ---------
    http://cfconventions.org/1.6#longitude-coordinate
    """
    name='lon' 
    long_name="longitude" 
    units="degrees_east"
    standard_name="longitude" 
    modulo=360.


#def _get_geoarray_cls(dims, globs=None):
#    """ look whether a particular pre-defined array matches the dimensions
#    """
#    if globs is None: globs = globals()
#    cls = None
#    for obj in globs.keys():
#        if isinstance(obj, globals()['GeoArray']):
#            if tuple(dims) == cls._dimensions:
#                cls = obj
#
#    return cls


#class CommonGeoArray(GeoArray):
#    #pass
#    _order = ('items','time','lat','lon','height','sample')
#    def __init__(self, values=None, *axes, **kwargs):
#        """
#        """
#        assert (len(axes) == 0 or len(kwargs) ==0), "cant provide axes both as kwargs and list"
#        assert self._dims is None or (len(axes) == self._dims or len(kwargs) == len(self._dims)), "dimension mismatch"
#        if len(kwargs) > 0:
#            for k in kwargs:
#                if k not in self._order:
#                    raise ValueError("unknown dimension, please provide as axes")
#            if self._dims is not None:
#                axes = [k,kwargs[k] for k in self._dims if k in kwargs]
#            else:
#                axes = [k,kwargs[k] for k in self._order if k in kwargs]
#
#        else:
#            if self._dims is not None:
#                assert tuple(ax.name for ax in axes) == self._dims, "dimension mismtach"
#
#        super(CommonGeoArray, self).__init__(values, axes)
#        for k in kwargs: self.setncattr(k, kwargs[k])
#    
#    @classmethod
#    def _constructor(cls, values, axes, **kwargs):
#        dims = tuple(ax.name for ax in axes)
#        class_ = _get_geoarray_cls(dims)
#        if class_ is not None:
#            obj = class_(values, *axes)
#        else:
#            obj = cls(values, *axes)
#        for k in kwargs: obj.setncattr(k, kwargs[k])
#        return obj

#
#class TimeSeries(GeoArray):
#    _dims = ('time',)
#
#class Map(GeoArray):
#    _dims = ('lat','lon')
#
#class TimeMap(GeoArray):
#    _dims = ('time','lat','lon')
#
#class Sample(GeoArray):
#    _dims = ('sample',)
