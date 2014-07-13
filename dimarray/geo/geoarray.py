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

    units = ""
    long_name = ""
    standard_name = ""

    grid_mapping = None

    def __init__(self, values=None, axes=None, 
            time=None, lat=None, lon=None, z=None, y=None, x=None, 
            dims=None, 
            standard_name="", long_name="", units="",
            #metadata = None,
            **kwargs):
        """ 
        Parameters
        ----------
        values : array-like
        axes : list of array-like or Axis objects, optional
        time, lat, lon, x, y, z : spatiotemporal coordinates, optional
           This form is a convenience that can be used instead of the `axes` 
           parameter, when applicable.
           It is not possible to mix-up these keyword arguments and 
           the `axes` parameter.
           Note it is assumed that the array shape follow the same order 
           as listed above. Otherwise use the `dims` parameter.
           Choose `lon` and `lat` for longitude and latitude (geodesic
           coordinate system), or `x` or `y` for other horizontal 
           coordinate systems such as in the projection plane.
        dims : sequence, optional
            sequence of axis names (dimensions)
            This is needed when axes are defined via keywords but 
            array-shape does not conform.
        standard_name, long_name, units : str, optional
            standard attributes according to the CF-1.4 netCDF 
            conventions. Will be added to metadata.
        **kwargs : keyword arguments, optional
            Passed to dimarray.DimArray

        See Also
        --------
        dimarray.DimArray : base class with no "geo" specificities
        dimarray.geo.Coordinate : base class for geo-axes
        dimarray.geo.Time 
        dimarray.geo.Latitude, dimarray.geo.Longitude
        dimarray.geo.Z, dimarray.geo.Y, dimarray.geo.X
        """
        keyword = (time is not None or lat is not None or lon is not None or x is not None or y is not None or z is not None)
        assert not (axes is not None and keyword), "can't input both `axes=` and keyword arguments!"
        assert not ((lon is not None or lat is not None) and (x is not None or y is not None)), "can't have both lon,lat and x,y horizontal coordinates"

        # construct the axes
        if keyword:
            axes = Axes()
            if time is not None: axes.append(time if isinstance(time, Axis) else Time(time, 'time'))
            if lat is not None: axes.append(lat if isinstance(lat, Axis) else Latitude(lat, 'lat'))
            if lon is not None: axes.append(lon if isinstance(lon, Axis) else Longitude(lon, 'lon'))
            if z is not None: axes.append(z if isinstance(z, Axis) else Z(z, 'z'))
            if y is not None: axes.append(y if isinstance(y, Axis) else Y(y, 'y'))
            if x is not None: axes.append(x if isinstance(x, Axis) else X(x, 'x'))
            if dims is not None:
                assert len(dims) == len(axes), "dims ({}) and axes ({}) \
                        lengths do not match".format(len(dims), len(axes))
                axes = [axes[nm] for nm in dims]

        #if metadata is not None:
        #    for k in metadata.keys():
        #        assert k not in kwargs, "{} was provided multiple times".format(k)
        #    kwargs.update(metadata) # TODO: make metadata a parameter in DimArray as well

        super(GeoArray, self).__init__(values, axes, **kwargs) # order dimensions

        # add metadata
        #if metadata is None: metadata = {}
        metadata = {}
        if units: metadata['units'] = units
        if standard_name: metadata['standard_name'] = standard_name
        if long_name: metadata['long_name'] = long_name

        # Do some guessing to define coordinates
        for i, ax in enumerate(self.axes):
            if isinstance(ax, Coordinate):
                pass
            if is_latitude(ax.name) or (hasattr(ax, 'standard_name') \
                    and ax.standard_name == 'latitude'):
                self.axes[i] = Latitude.from_axis(ax)
            if is_longitude(ax.name) or (hasattr(ax, 'standard_name') \
                    and ax.standard_name == 'longitude'):
                self.axes[i] = Longitude.from_axis(ax)
            if is_time(ax.name):
                self.axes[i] = Time.from_axis(ax)
            # 'x', 'y', 'z' are too general to be used.
            if ax.name == 'x' or (hasattr(ax, 'standard_name') \
                    and ax.standard_name == 'projection_x_coordinate'):
                self.axes[i] = X.from_axis(ax)
            if ax.name == 'y' or (hasattr(ax, 'standard_name') \
                    and ax.standard_name == 'projection_y_coordinate'):
                self.axes[i] = Y.from_axis(ax)
            if ax.name in ('z','height','depth'):
                self.axes[i] = Z.from_axis(ax)

        # check 

    def __repr__(self): return super(GeoArray, self).__repr__().replace("dimarray","geoarray")
    def __print__(self): return super(GeoArray, self).__print__().replace("dimarray","geoarray")
    def __str__(self): return super(GeoArray, self).__str__().replace("dimarray","geoarray")

#    def _plot2D(self, function, *args, **kwargs):
#        import matplotlib.pyplot as plt
#        import dimarray.geo as geo
#
#        # transform the data prior to plotting?
#        # (has additional ability of reading own attributes)
#        transform = kwargs.pop('transform', None)
#        if transform:
#            self = geo.transform(self, to_grid_mapping=transform) # transform the data
#
#        # use a canvas suited to the coordinate system?
#        if ax not in kwargs:
#            if hasattr(self, 'grid_mapping'):
#                try:
#                    projection = get_grid_mapping(self.grid_mapping)
#                except:
#                    projection = None
#
#            # input has priority
#            projection = kwargs.pop('projection', projection)
#
#            if projection is not None:
#                kwargs['projection'] = projection
#
#        else:
#            assert projection is None, "can't have both ax and projection"
#
#        super(GeoArray, self)._plot2D(function, *args, **kwargs)




# define Latitude and Longitude axes
class Coordinate(Axis):
    """ Axis with a geophysical meaning, following the netCDF CF conventions
    """
    __metadata_include__  = ['standard_name', 'long_name', 'units'] # may be defined as class attributes
    def __init__(self, values, name=None, dtype=float, standard_name="", long_name="", units="", **kwargs):

        #metadata = dict()
        metadata = self._metadata
        #metadata.update(self.__class__.__dict__) # add class metadata to metadata

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

    def __repr__(self):
        return super(Coordinate, self).__repr__()+" (Coord-{})".format(self.__class__.__name__)


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
