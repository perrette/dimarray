""" Subclass of DimArray specialized for geo-applications

==> dimensions always remain sorted: ('items','time','lat','lon','height','sample')
==> subset of methods with specific plotting behaviour: e.g. TimeSeries, Map
"""
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

    def __init__(self, values=None, axes=None, time=None, lat=None, lon=None, x=None, y=None, **kwargs):
        """ 
        """
        keyword = (time is not None or lat is not None or lon is not None)
        assert not (axes is not None and keyword), "can't input both `axes=` and keyword arguments!"

        # construct the axes
        if keyword:
            axes = Axes()
            if time is not None: axes.append(Time(time, 'time'))
            if lat is not None: axes.append(Latitude(lat, 'lat'))
            if lon is not None: axes.append(Longitude(lon, 'lon'))
            if x is not None: axes.append(X(x, 'x'))
            if y is not None: axes.append(Y(y, 'y'))

        super(GeoArray, self).__init__(values, axes, **kwargs) # order dimensions

        # add weight on latitude
        for i, ax in enumerate(self.axes):
            if is_latitude(ax.name) and not isinstance(ax, GeoAxis):
                self.axes[i] = Latitude.from_axis(ax)
            if is_longitude(ax.name) and not isinstance(ax, GeoAxis):
                self.axes[i] = Longitude.from_axis(ax)
            if is_time(ax.name) and not isinstance(ax, GeoAxis):
                self.axes[i] = Time.from_axis(ax)
            if ax.name == 'x':
                self.axes[i] = X.from_axis(ax)
            if ax.name == 'y':
                self.axes[i] = Y.from_axis(ax)

    def __repr__(self): return super(GeoArray, self).__repr__().replace("dimarray","geoarray")
    def __print__(self): return super(GeoArray, self).__print__().replace("dimarray","geoarray")
    def __str__(self): return super(GeoArray, self).__str__().replace("dimarray","geoarray")

    # add projection features
    def project(self, grid_mapping, x=None, y=None, from_grid_mapping=None):
        """ Project to a new coordinate system and interpolate values onto a new axis

        Parameters
        ----------
        x, y : array-like, optional
            new coordinates to interpolate the array on
            will be deduced as min and max of new coordinates if not provided

        from_grid_mapping : str or dict or cartopy.crs.CRS instance, optional
            original grid mapping, to be provided only when no grid_mapping attribute is defined
            or when the axes are something else than Longitude and Latitude
        """ 
        pass


# define Latitude and Longitude axes
class GeoAxis(Axis):
    @classmethod
    def from_axis(cls, ax):
        " define a GeoAxis from an Axis object "
        return cls(ax.name, ax.values, **ax._metadata)

    def __repr__(self):
        return super(self.__class__, self).__repr__()+" (Geo{})".format(self.__class__.__name__)

class Latitude(GeoAxis):
    def __init__(self, *args, **kwargs):
        kwargs['weights'] = lambda x: np.cos(np.radians(x))
        if not 'dtype' in kwargs: kwargs['dtype'] = np.float
        super(self.__class__, self).__init__(*args, **kwargs)
         
class Longitude(GeoAxis):
    def __init__(self, *args, **kwargs):
        kwargs['modulo'] = 360
        if not 'dtype' in kwargs: kwargs['dtype'] = np.float
        super(self.__class__, self).__init__(*args, **kwargs)

# To handle projection systems
class X(GeoAxis):
    standard_name = "projection_x_coordinate";
    long_name = "x distance on the projection plane from the origin";
    #units = "km";

class Y(GeoAxis):
    standard_name = "projection_y_coordinate";
    long_name = "y distance on the projection plane from the origin";

class Time(GeoAxis):
    " Provided the appropriate metadata, should handle date conversions"
    pass

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
