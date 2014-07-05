""" Subclass of DimArray specialized for geo-applications

==> dimensions always remain sorted: ('items','time','lat','lon','height','sample')
==> subset of methods with specific plotting behaviour: e.g. TimeSeries, Map

References
----------
    http://cfconventions.org/1.4
    http://cfconventions.org/1.6 (in track-change mode w.r.t CF-1.4)
"""
from collection import OrderedDict as odict
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
        keyword = (time is not None or lat is not None or lon is not None)
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
        if metadata is None: metadata = {}
        if units: metadata['units'] = units
        if standard_name: metadata['standard_name'] = standard_name
        if long_name: metadata['long_name'] = long_name

        # Transform axes to "geo" axes
        for i, ax in enumerate(self.axes):
            if isinstance(ax, Coordinate):
                pass
            if is_latitude(ax.name):
                self.axes[i] = Latitude.from_axis(ax)
            if is_longitude(ax.name):
                self.axes[i] = Longitude.from_axis(ax)
            if is_time(ax.name):
                self.axes[i] = Time.from_axis(ax)
            if ax.name == 'x':
                self.axes[i] = X.from_axis(ax)
            if ax.name == 'y':
                self.axes[i] = Y.from_axis(ax)
            if ax.name == 'z':
                self.axes[i] = Z.from_axis(ax)

        # check 

    def __repr__(self): return super(GeoArray, self).__repr__().replace("dimarray","geoarray")
    def __print__(self): return super(GeoArray, self).__print__().replace("dimarray","geoarray")
    def __str__(self): return super(GeoArray, self).__str__().replace("dimarray","geoarray")

    def transform(self, to_grid_mapping, from_grid_mapping=None, x1=None, y1=None, horizontal_coordinates=None):
        """ Transform to a new coordinate system and interpolate values onto a new regular grid

        Parameters
        ----------
        to_grid_mapping : str or dict or cartopy.crs.CRS instance
            grid mapping onto which the transformation should be done
        from_grid_mapping : idem, optional
            original grid mapping, to be provided only when no grid_mapping 
            attribute is defined or when the axes are something else than 
            Longitude and Latitude
        x1, y1 : array-like (1-D), optional
            new coordinates to interpolate the array on
            will be deduced as min and max of new coordinates if not provided
        horizontal_coordinates : sequence with 2 str, optional
            provide horizontal coordinates by name if not an instance of 
            Latitude, Longitude, X or Y

        Return
        ------
        transformed : GeoArray
            new GeoArray transformed
        """ 
        from projection import Geodetic, transform  # avoid loading the projection module since it's quite heavy
        from dimarray.compat.basemap import interp

        # find horizontal coordinates
        if horizontal_coordinates: 
            assert len(horizontal_coordinates) == 2, "horizontal_coordinates must be a sequence of length 2"
            x0nm, y0nm = horizontal_coordinates
            x0 = self.axes[x0nm]
            y0 = self.axes[y0nm]

        else:
            xs = filter(lambda x: isinstance(x, X), self.axes)
            ys = filter(lambda x: isinstance(x, Y), self.axes)
            longs = filter(lambda x: isinstance(x, Longitude), self.axes)
            lats = filter(lambda x: isinstance(x, Latitude), self.axes)

            if len(xs) == 1:
                x0 = xs[0]
            elif len(longs) == 1:
                x0 = longs[0]
            else:
                raise Exception("Could not find X-coordinate among GeoArray axes")

            if len(ys) == 1:
                y0 = ys[0]
            elif len(lats) == 1:
                y0 = lats[0]
            else:
                raise Exception("Could not find Y-coordinate among GeoArray axes")

        # determine grid mapping
        if from_grid_mapping is None:

            # first check if grid_mapping attribute is defined
            if hasattr(self, 'grid_mapping'):
                from_grid_mapping = self.grid_mapping

            # otherwise check whether the Coordinates are Geodetic (long/lat)
            else:
                if isinstance(x0, Longitude) and isinstance(y0, Latitude):
                    from_grid_mapping = Geodetic()

        # double-check
        if from_grid_mapping is None:
            raise Exception("provide from_grid_mapping or define grid_mapping attribute")

        # Create a 2-D grid with horizontal coordinates
        x0_2d, y0_2d = np.meshgrid(x0.values, y0.values)

        # Transform coordinates 
        x1_2d, y1_2d = transform(x0_2d, x0_2d, from_grid_mapping, to_grid_mapping)

        # Interpolate onto a new regular grid while roughly conserving the steps
        if x1 is None:
            x1 = np.linspace(x1_2d.min(), x1_2d.max(), x1_2d.shape[1])
        if y1 is None:
            y1 = np.linspace(y1_2d.min(), y1_2d.max(), y1_2d.shape[0])

        x1r_2d, y1r_2d = np.meshgrid(x1, y1) # regular 2-D grid

        if self.ndim == 2:
            newvalues = interp(self.values, x1_2d, y1_2d, x1r_2d, y1r_2d)

        else:
            # first reshape to 3-D, flattening everything except vertical coordinates
            # TODO: optimize by computing and re-using weights?
            obj = self.group((x0.name, x1.name), reverse=True, insert=0)  
            newvalues = []
            for k, suba in obj.iter(axis=0): # iterate over the first dimension
                newval = interp(suba.values, x1_2d, y1_2d, x1r_2d, y1r_2d)
                newvalues.append(newval)

            # stack the arrays together
            np.stack(newvalues)


# define Latitude and Longitude axes
class Coordinate(Axis):
    """ Axis with a geophysical meaning, following the netCDF CF conventions
    """
    def __init__(self, values, name=None, dtype=float, standard_name="", long_name="", units="", **kwargs):
        #if metadata is None: metadata = {}

        if hasattr(self, "name"): # class attribute?
            name = self.name

        metadata = dict()
        metadata.update(self.__class__.__dict__) # add class metadata to metadata

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

        super(self.__class__, self).__init__(values, name, dtype=dtype, **metadata) 

    @classmethod
    def from_axis(cls, ax):
        " define a Coordinate from an Axis object "
        return cls(ax.name, ax.values, **ax._metadata)

    def __repr__(self):
        return super(self.__class__, self).__repr__()+" (Geo{})".format(self.__class__.__name__)

class Time(Coordinate):
    """ Time coordinate

    Reference
    ---------
    http://cfconventions.org/1.6#time-coordinate
    """
    name = "time"

class Latitude(Coordinate):
    """ Latitude Coordinate

    Reference
    ---------
    http://cfconventions.org/1.6#latitude-coordinate
    """
    name = 'lat', 
    long_name = "latitude", 
    units = "degrees_north", 
    standard_name = "latitude", 
    weights = lambda x: np.cos(np.radians(x))
         
class Longitude(Coordinate):
    """ Longitude Coordinate

    Reference
    ---------
    http://cfconventions.org/1.6#longitude-coordinate
    """
    name='lon', 
    long_name="longitude", 
    units="degrees_east", 
    standard_name="longitude", 
    modulo=360., 

# To handle projection systems
class X(Coordinate):
    """ X-horizontal coordinate on the projection plane

    Reference
    ---------
    http://cfconventions.org/1.6#grid-mappings-and-projections
    """
    name = 'x',
    long_name = "x distance on the projection plane from the origin";
    standard_name = "projection_x_coordinate";
    #units = "km";

class Y(Coordinate):
    """ Y-horizontal coordinate on the projection plane

    Reference
    ---------
    http://cfconventions.org/1.6#grid-mappings-and-projections
    """
    name = 'y'
    long_name = "y distance on the projection plane from the origin";
    standard_name = "projection_y_coordinate";

class Z(Coordinate):
    """ Vertical coordinate

    Reference
    ---------
    http://cfconventions.org/1.6#vertical-coordinate "
    """
    name = 'z'

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
