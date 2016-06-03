""" Wrapper around cartopy's transformation using standard netCDF parameters (CF-1.4, 1.6)

The aim is to be able to read and transform coordinate systems from a netCDF file looking like:
    var1:
        grid_mapping = "mapping"
        long_name = ...
        etc...

    mapping:
        grid_mapping_name = ...
        latitude_of_projection_origin = ...
        ...

The classes will inherit from cartopy's projection classes but 
be initialized with netCDF4 projections names.

References
----------
PROJ.4 Projections: 
    http://www.remotesensing.org/geotiff/proj_list
    http://trac.osgeo.org/proj/wiki/GenParms
NetCDF Conventions (CF):
    http://cfconventions.org
    http://cfconventions.org/1.4
    http://cfconventions.org/1.6 (in track-change mode w.r.t CF-1.4)
    E.g.: http://cfconventions.org/1.4.html#grid-mappings-and-projections
NetCDF Conventions in Java: 
    https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/reference/StandardCoordinateTransforms.html

"""
import sys
import inspect
import warnings
import math
import numpy as np
import dimarray.compat.cartopy_crs as ccrs

from collections import OrderedDict as odict

from dimarray.tools import format_doc, file_an_issue_message

class Globe(ccrs.Globe):

    # just info
    _proj4_def = """+datum=datum +ellps=ellipsoid +a=semi_major_axis
                    +b=semi_minor_axis +f=flattening +rf=inverse_flattening
                    +towgs84=towgs84 +nadgrids=nadgrids """

    def __init__(self, datum=None, ellipsoid='WGS84',
                 semi_major_axis=None, semi_minor_axis=None,
                 flattening=None, inverse_flattening=None,
                 towgs84=None, nadgrids=None):

        super(Globe, self).__init__( 
                datum=datum, ellipse=ellipsoid,
                semimajor_axis=semi_major_axis, 
                semiminor_axis=semi_minor_axis, 
                flattening=flattening, inverse_flattening=inverse_flattening, 
                towgs84=towgs84, nadgrids=nadgrids)

    @classmethod
    def from_proj4(cls, proj4_params):
        """ initialize from a dictionary of PROJ.4 parameters
        """
        table = dict(_parse_proj4(cls._proj4_def))
        cf_params = {table[k]:val for k,val in proj4_params.iteritems()}
        return Globe(**cf_params)


class CF_CRS(object):
    """ Projections initialized with CF-compatible parameters
    """
    grid_mapping_name = None

    # can define a dictioary containing standard metadata for the coordinates
    # only standard_name is enforced by CF
    _x_metadata = dict(
        name = 'x',
        long_name = "x distance on the projection plane from the origin",
        standard_name = "projection_x_coordinate",
        units = "meters", # at least in cartopy
        )

    _y_metadata = dict(
        name = 'y',
        long_name = "y distance on the projection plane from the origin",
        standard_name = "projection_y_coordinate",
        units = "meters", # at least in cartopy
        )
    _z_metadata = None

#
# Now the actual Projection classes
#

class LatitudeLongitude(CF_CRS, ccrs.Geodetic):
    """ Same as cartopy.crs.Geodetic but is initialized via CF parameters
    """
    grid_mapping_name = 'latitude_longitude'

    _proj4_def = '+proj=lonlat +proj=longlat' # +lon_0=longitude_of_prime_meridian' is gone??
    # _proj4_def = '+proj=lonlat +proj=longlat +proj=eqc +lon_0=longitude_of_prime_meridian' # several possibilities

    _x_metadata = dict(
        name = 'lon',
        long_name = "longitude",
        units = "degrees_east",
        standard_name = "longitude",
        )

    _y_metadata = dict(
        name = 'lat',
        long_name = "latitude",
        units = "degrees_north",
        standard_name = "latitude",
        )

    # def __init__(self,  longitude_of_prime_meridian = 0.0, **kwargs):
    def __init__(self,  longitude_of_prime_meridian = 0.0, **kwargs):
        # TODO: check out longitude_of_prime_meridian and find a way of including it using Cartopy's projections
        if longitude_of_prime_meridian != 0:
            warnings.warn("longitude_of_prime_meridian={} parameter is ignored in LatitudeLongitude transformation".repr(longitude_of_prime_meridian))
        globe = Globe(**kwargs)
        # for now not possible because PlateCarree initialization depends on whether globe is None or not.
        ccrs.Geodetic.__init__(self, globe)


class Stereographic(CF_CRS, ccrs.Stereographic):

    grid_mapping_name = "stereographic"

    # initialize from PROJ.4
    _proj4_def = """+proj=stere
                +lat_0=latitude_of_projection_origin
                +lon_0=longitude_of_projection_origin
                +x_0=false_easting
                +y_0=false_northing
                +k_0=scale_factor_at_projection_origin
                """

    def __init__(self, 
            latitude_of_projection_origin=0.0, 
            longitude_of_projection_origin=0.0, 
            false_easting=0.0, false_northing=0.0, 
            scale_factor_at_projection_origin=None, 
            **kwargs):

        globe = Globe(**kwargs)

        # standard_parallel is not in the CF-conventions
        ccrs.Stereographic.__init__(self, 
                central_longitude=longitude_of_projection_origin,
                central_latitude=latitude_of_projection_origin,
                false_northing=false_northing, false_easting=false_easting,
                scale_factor=scale_factor_at_projection_origin,
                globe=globe)

#
#
#
class PolarStereographic(Stereographic):

    grid_mapping_name = "polar_stereographic"

    _proj4_def = """+proj=stere
                +lon_0=straight_vertical_longitude_from_pole
                +lat_0=latitude_of_projection_origin
                +x_0=false_easting
                +y_0=false_northing
                +lat_ts=standard_parallel
                +k_0=scale_factor_at_projection_origin
            """

    # correspondence table between netCDF and Cartopy parameters
    def __init__(self, latitude_of_projection_origin=90., 
            straight_vertical_longitude_from_pole=0.0, 
            false_easting=0.0, false_northing=0.0, 
            scale_factor_at_projection_origin=None, 
            standard_parallel=None, **kwargs):

        if latitude_of_projection_origin not in (-90, 90):
            msg = "latitude_of_projection_origin must be -90 or +90"
            raise ValueError(msg)

        globe = Globe(**kwargs)

        ccrs.Stereographic.__init__(self, 
                central_latitude=latitude_of_projection_origin,
                central_longitude=straight_vertical_longitude_from_pole,
                false_northing=false_northing, false_easting=false_easting,
                scale_factor=scale_factor_at_projection_origin,
                true_scale_latitude=standard_parallel,
                globe=globe)


class RotatedPole(CF_CRS, ccrs.RotatedPole):

    grid_mapping_name = "rotated_latitude_longitude"

    _proj4_def = """
        +proj=ob_tran
        """
    # other arguments are not used here
    #    +o_proj=latlon
    #    +o_lon_p=0
    #    +o_lat_p=grid_north_pole_latitude
    #    +lon_0=180 + grid_north_pole_longitude
    #    +to_meter=0.017453292519943295

    _x_metadata = dict(
            name = 'rlon',
            standard_name = 'grid_longitude',
            long_name = "longitude in rotated pole grid",
            units = "degrees",
            )

    _y_metadata = dict(
            name = 'rlat',
            standard_name = 'grid_latitude',
            long_name = "latitude in rotated pole grid",
            units = "degrees",
            )

    def __init__(self, 
            grid_north_pole_longitude=0.0, 
            grid_north_pole_latitude=90.0,
            **kwargs):

        globe = Globe(**kwargs)

        ccrs.RotatedPole.__init__(self, pole_latitude=grid_north_pole_latitude,
                pole_longitude=grid_north_pole_longitude,
                globe=globe)


class TransverseMercator(CF_CRS, ccrs.TransverseMercator):
    grid_mapping_name = "transverse_mercator"

    _proj4_def = """+ellps=ellipsoid +proj=tmerc 
                    +lon_0=longitude_of_central_meridian 
                    +lat_0=latitude_of_projection_origin 
                    +k=scale_factor_at_central_meridian 
                    +x_0=false_easting
                    +y_0=false_northing
                    +units=meters
                 """

    def __init__(self, 
            longitude_of_central_meridian=0.0,
            latitude_of_projection_origin=0.0,
            false_northing=0.0,
            false_easting=0.0,
            scale_factor_at_central_meridian=1.0,
            **kwargs):

        globe = Globe(**kwargs)

        ccrs.TransverseMercator.__init__(self, 
                central_longitude=longitude_of_central_meridian, 
                central_latitude=latitude_of_projection_origin,
                false_easting=false_easting, false_northing=false_northing,
                scale_factor=scale_factor_at_central_meridian, 
                globe=globe)

#
# Perform projections purely based on PROJ.4 parameters
#
class Proj4(ccrs.CRS):
    """ Inherit from cartopy.crs.CRS, initialized from a PROJ.4 string

    Notes
    -----
    :meth:`transform_vectors` method will not work, because no x_limits 
    and y_limits are defined.
    """
    def __init__(self, proj4_init=None, **proj4_params):
        """ initialize a CRS instance based on PROJ.4 parameters

        Parameters
        ----------
        proj4_init : PROJ.4 string, optional
        **proj4_params : PROJ.4 parameters as keyword arguments, optional

        Examples
        --------
        >>> prj = Proj4("+ellps=WGS84 +proj=stere +lat_0=90.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +lat_ts=71.0")
        """
        assert proj4_init is not None or len(proj4_params) > 0, "no argument provided"
        assert not (proj4_init is not None and len(proj4_params) > 0), "must provide EITHER a string of key-word arguments"
        if len(proj4_params) == 0:
            proj4_params = _parse_proj4(proj4_init) # key, value pair
            proj4_params = odict(proj4_params)

        # filter globe parameters
        globe_params = {k:proj4_params.pop(k) for k in proj4_params.keys() \
                if "+{}=".format(k) in Globe._proj4_def}

        globe = Globe.from_proj4(globe_params)

        # initialize CRS instance
        super(Proj4, self).__init__(proj4_params, globe=globe)


def _parse_proj4(proj4_init):
    """ convert proj4 string to (key, value) pairs
    """
    assert isinstance(proj4_init, str), "must be str"

    tmp = [arg.split('=') for arg in proj4_init.split()] # +key, value pair
    msg = "invalid PROJ.4 str, must be of the form +param=value +param2=value2, got {}".format(proj4_init)

    try:
        pkeys, values = zip(*tmp)
    except ValueError:
        if "+no_defs" in proj4_init: 
            msg += "(e.g. try removing +no_defs)"
        raise ValueError(msg)

    try:
        keys = [k.split('+')[1] for k in pkeys] # remove the +
    except IndexError:
        raise ValueError(msg)

    # convert type
    values = list(values)
    for i, v in enumerate(values):
        try:
            values[i] = float(v)
        except:
            pass

    return zip(keys, values)


#
# Inspect the CF_CRS instances
#
def _get_cf_crs_classes(predicate=None):
    """ return one or several CF_CRS subclass
    predicate : function which accept a CF_CRS class as argument
    """
    if predicate is None:
        predicate = lambda x: x.grid_mapping_name is not None 

    current = sys.modules[__name__] # current module
    res = inspect.getmembers(current , lambda x: np.issubclass_(x, CF_CRS) and predicate(x))
    if len(res) > 0:
        classes = zip(*res)[1] # res is a list of (class name, class)
        classes = _sort_classes(classes) # sort classes, the first one if the most relevant
    else:
        classes = []
    return classes

def _sort_classes(classes):
    """ sort classes by priority """
    # if several classes did work, then choose the most specialized class
    # e.g. Stereo and PolarStereo 
    def cmp_(cls0 , cls1):
        if np.issubclass_(cls0, cls1):
            return -1
        elif np.issubclass_(cls1, cls0):
            return 1
        else:
            return 0 # do not change order
    classes = list(classes)
    classes.sort(cmp=cmp_)
    return classes

def _get_cf_crs_class(grid_mapping_name):
    """ return one grid mapping class based on grid_mapping_name
    """
    predicate = lambda x: x.grid_mapping_name == grid_mapping_name
    found = _get_cf_crs_classes(predicate)

    if len(found) == 0:
        found_all = _get_cf_crs_classes()
        accepted = [tup.grid_mapping_name for tup in found_all]

        msg = "Unknown projection: {}. Currently accepted projections: {}"
        msg = msg.format(grid_mapping_name, ", ".join(accepted))
        msg += "\n"+file_an_issue_message()
        raise ValueError(msg)

    assert len(found) == 1

    return found[0] # 0: class name, 1: class

def get_crs(grid_mapping):
    """ Get a cartopy's CRS instance from a variety of key-word parameters

    Parameters
    ----------
    grid_mapping : str or dict or cartopy.crs.CRS instance
        See notes below.

    Returns
    -------
    cartopy.crs.CRS instance 

    Notes
    -----
    A grid mapping can be defined in one of the following ways:

    - providing a cartopy.crs.CRS instance directly
    - provide a cartopy.crs.CRS subclass name, for initialization with 
      default parameters
    - providing a dictionary of netCDF-conform parameters (CF-1.4)
    - provide a PROJ.4 string, with parameters preceded by '+' (EXPERIMENTAL)

    References
    ----------
    - CF-conventions : http://cfconventions.org
        See `Appendix F on grid mapping <http://cfconventions.org/1.4.html#appendix-grid-mappings>`_

    - PROJ.4 projections : http://www.remotesensing.org/geotiff/proj_list

    - PROJ.4 parameters : https://trac.osgeo.org/proj/wiki/GenParms

    - Cartopy projections: 
        http://scitools.org.uk/cartopy/docs/latest/crs/projections.html

    Examples
    --------

    Import Cartopy's crs module

    >>> import dimarray.compat.cartopy_crs as ccrs

    Longitude / Latitude coordinates

    >>> grid_mapping = ccrs.Geodetic() # cartopy
    >>> grid_mapping = "geodetic"  # cartopy class name
    >>> grid_mapping = {'grid_mapping_name':'latitude_longitude'} # CF-1.4

    Other coordinates systems onto which lon and lat can be projected onto

    >>> from dimarray.geo.crs import get_crs

    North Polar Stereographic Projection over Greenland

    ... with Cartopy Stereographic class, and all parameters
        (note true_scale_latitude only makes sense for polar stereo)

    >>> globe = ccrs.Globe(ellipse='WGS84')  # it's the default anyway
    >>> crs = ccrs.Stereographic(  
    ...     central_latitude  = 90.,  # center of projection 
    ...     central_longitude = -39., # center of projection
    ...     true_scale_latitude = 71., # only makes sense for polar stereo
    ...     false_easting = 0.,  # default offset to express x w.r.t 
    ...     false_northing = 0., # default offset to express y
    ...     globe=globe)  # cartopy.crs.CRS instance
    >>> crs.transform_point(70, -40, ccrs.Geodetic())
    (24969236.85758362, 8597597.732836112)

    ... with Cartopy NorthPolarStereo class (central_latitude=90. by default)

    >>> crs = ccrs.NorthPolarStereo(central_longitude=-39., true_scale_latitude=71.)  
    >>> crs.transform_point(70, -40, ccrs.Geodetic())
    (24969236.85758362, 8597597.732836112)

    ... Using CF-1.4 and later conventions, intended for netCDF files.

    >>> grid_mapping = dict(
    ...     grid_mapping_name = 'polar_stereographic', 
    ...     latitude_of_projection_origin = 90., 
    ...     straight_vertical_longitude_from_pole = -39.,
    ...     standard_parallel = 71.,
    ...     false_easting = 0.,  # default
    ...     false_northing = 0., 
    ...     ellipsoid = 'WGS84', 
    ...     )  # CF-1.4
    >>> crs = get_crs(grid_mapping)
    >>> crs.transform_point(70, -40, ccrs.Geodetic())
    (24969236.85758362, 8597597.732836112)

    ... PROJ.4 equivalent, with all parameters

    >>> proj4_init = "+ellps=WGS84 +proj=stere +lat_0=90.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +lat_ts=71.0"
    >>> crs = get_crs(proj4_init)
    >>> crs.transform_point(70, -40, ccrs.Geodetic())
    (24969236.85758362, 8597597.732836112)
    """
    if isinstance(grid_mapping, ccrs.CRS):
        return grid_mapping

    if isinstance(grid_mapping, dict):
        grid_mapping = grid_mapping.copy()
        try:
            grid_mapping_name = grid_mapping.pop('grid_mapping_name')
        except KeyError:
            raise ValueError("grid_mapping_name not present")

        cls = _get_cf_crs_class(grid_mapping_name)
        coord_sys = cls(**grid_mapping)

    elif isinstance(grid_mapping, str):

        # special case: PROJ.4 string?
        if '+' in grid_mapping:
            coord_sys = Proj4(grid_mapping)

        # common case: Cartopy class name
        else:
            members = inspect.getmembers(ccrs, lambda x: np.issubclass_(x, ccrs.CRS) and x.__name__.lower() == grid_mapping.lower())
            if len(members) == 0:
                raise ValueError("class "+grid_mapping+" not found")
            else:
                assert len(members) == 1, "more than one match, bug: check case"
                cls = members[0][1]
                #cls = getattr(ccrs, grid_mapping)
            coord_sys = cls()

    else:
        msg = 'grid_mapping: must be str or dict or cartopy.crs.CRS instance, got: {}'.format(grid_mapping)
        raise TypeError(msg)

    # just checking
    assert isinstance(coord_sys, ccrs.CRS), 'something went wrong'

    return coord_sys
