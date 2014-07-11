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
import cartopy
import cartopy.crs as ccrs
#from cartopy.crs import Projection, Globe
import dimarray.compat.cartopy as compat_crs

from dimarray.decorators import format_doc
from dimarray.info import file_an_issue_message

# check cartopy version
def _check_cartopy_version():
    M = cartopy.__version__.split('.')[0]
    m = cartopy.__version__.split('.')[1]
    if int(M) == 0 and int(m) < 11:
        warnings.warn('Projections were only tested for cartopy versions 0.11.x')

    # update Stereographic proejctions
    if int(M) < 999: # update with proper cartopy version if the pull request 
        # is accepted
        ccrs.Stereographic = compat_crs.Stereographic
        ccrs.NorthPolarStereo = compat_crs.NorthPolarStereo
        ccrs.SouthPolarStereo = compat_crs.SouthPolarStereo

_check_cartopy_version()


#
# Util functions used in Globe and Projection
#
def _assert_alternate_params(kwargs, params):
    """ check for redundant parameters
    """
    matching = []
    for k in kwargs:
        if k in params:
            matching.append(k)

    if len(matching) > 1:
        raise ValueError("Redundant parameter definition, please use only one of: "+", ".join(matching))

def _error_message(error, table):
    " make a typical error message when initializing a Cartopy projection fails"
    msg = "Cartopy Exception:\n\n"
    msg += error.message
    msg += "\n\n----------------------------------------------------"
    msg += "\n"
    msg += "\nProblem when initializing Cartopy's projection class"
    msg += "\nSee corresponding Cartopy documentation for more info"
    msg += "\nThe table of correspondence between netCDF and Cartopy"
    msg += "\nparameters is {}.".format(table)
    msg += "\n"
    msg += "\n"+file_an_issue_message()
    return msg


class CF_CRS(object):
    """ Projection class to map Cartopy's implementation onto netCDF parameter names
        
    To be subclassed
    """
    # CF to cartopy format
    _cartopy_globe = dict(datum='datum', ellipsoid='ellipse', 
            semi_major_axis='semimajor_axis', 
            semi_minor_axis='semiminor_axis', 
            flattening='flattening', inverse_flattening='inverse_flattening', 
            towgs84='towgs84', nadgrids='nadgrids')

    _cartopy = {} # correspondence between netCDF and Cartopy names

    # proj4 and CF format
    _proj4_def = " "

    _proj4_def_globe = """  +datum=datum 
                        +ellps=ellipsoid 
                        +a=semi_major_axis 
                        +b=semi_minor_axis
                        +f=flattening
                        +rf=inverse_flattening
                        +towgs84=towgs84
                        +nadgrids=nadgrids
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

    def __init__(self, **kwargs):
        """ Initialize a projection class with CF-1.4 conforming parameters

        See class's `_cartopy` and `_cartopy_globe` attribute for correspondence 
        between CF-1.4 and Cartopy parameters
        """
        assert isinstance(self, ccrs.CRS), "dimarray.geo.CF_CRS class must be subclassed with another cartopy.crs.CRS instance"

        self._check_params(kwargs)

        cartopy_params = {}
        globe_params = {} # also Cartopy format
        #globe_kwargs = {}
        for k in kwargs.keys(): # CF-1.4 format
            if k in self._cartopy.keys():
                cartopy_params[self._cartopy[k]] = kwargs[k]
            elif k in self._cartopy_globe.keys():
                globe_params[self._cartopy_globe[k]] = kwargs[k]
            else:
                msg = "{} unexpected. Accepted parameters: {}".format(k, self._cartopy.keys()+self._cartopy_globe.keys())
                raise ValueError(msg)

        # Initialize a globe
        try:
            globe = ccrs.Globe(**globe_params)
            cartopy_params['globe'] = globe
        except Exception as error:
            msg = _error_message(error, self._cartopy_globe)
            raise error.__class__(msg)

        # Initialize cartopy coordinate system
        try:
            # find cartopy.crs.CRS ancestor:
            CRSproj = self.__class__
            while np.issubclass_(CRSproj, CF_CRS):
                CRSproj = filter(lambda x : np.issubclass_(x, ccrs.CRS), CRSproj.__bases__)[0]
            CRSproj.__init__(self, **cartopy_params)
            #super(CRSproj, self)(**cartopy_params)

        except Exception as error:
            msg = _error_message(error, self._cartopy)
            raise error.__class__(msg)

        # check that init went well
        assert isinstance(self, ccrs.CRS)
        assert self.proj4_init is not None
        assert self.proj4_params is not None

        # initialize dict of cf_params
        self.cf_params = self._proj4_to_cf_params(self.proj4_params)

    def _check_params(self, kwargs):
        pass

    @classmethod
    def _proj4_to_cf_params(cls, proj4_params):
        # get CF-parameters from PROJ.4, based on parsing cls._proj4_def
        # works well for standard cases
        # but can be subclassed for complicated cases
        proj4_def = cls._proj4_def + " " + cls._proj4_def_globe
        table = dict(_parse_proj4(proj4_def))

        # Convert PROJ.4 into CF parameters using the lookup table
        cf_params = {'grid_mapping_name':cls.grid_mapping_name}

        for k, val in proj4_params.iteritems(): # cartopy property
            if k == 'proj': continue
            nm_cf = table[k]
            #if val == 'IGNORE': continue
            if nm_cf == 'IGNORE': continue
            cf_params[nm_cf] = val

        return cf_params

    @classmethod
    def from_proj4(cls, proj4_params):
        """ initialize instance based on a dict of proj4_parameters
        """
        cf_params = cls._proj4_to_cf_params(proj4_params)
        del cf_params['grid_mapping_name']
        return cls(**cf_params)


#
# NOTE: the grid_mapping_name is 'longitude_latitude' and not 'geodetic'
#
class Geodetic(CF_CRS, ccrs.Geodetic):
    """ Same as cartopy.crs.Geodetic but accept netCDF parameters as arguments
    """
    grid_mapping_name = 'latitude_longitude'
    _proj4_def = '+proj=lonlat'
    _cartopy = {} # no parameter apart from "globe"

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


class Stereographic(CF_CRS, ccrs.Stereographic):

    grid_mapping_name = "stereographic"

    # correspondence cartopy between netCDF and Cartopy parameters
    _cartopy = dict(
            longitude_of_projection_origin = 'central_longitude',
            latitude_of_projection_origin = 'central_latitude',
            false_easting = 'false_easting',
            false_northing = 'false_northing',
            scale_factor_at_projection_origin = 'scale_factor',
            #standard_parallel = 'true_scale_latitude', # this is not the preferred way according to CF-1.4
            )

    _proj4_def = """+proj=stere
                +lon_0=longitude_of_projection_origin
                +lat_0=latitude_of_projection_origin
                +x_0=false_easting
                +y_0=false_northing
                +k_0=scale_factor_at_projection_origin
                +lat_ts=standard_parallel
                """

#
#
#
class PolarStereographic(Stereographic):

    grid_mapping_name = "polar_stereographic"

    # correspondence table between netCDF and Cartopy parameters
    _cartopy = dict(
            straight_vertical_longitude_from_pole = 'central_longitude',  # preferred?
            #longitude_of_projection_origin = 'central_longitude', # also accepted
            latitude_of_projection_origin = 'central_latitude',
            standard_parallel = 'true_scale_latitude',
            scale_factor_at_projection_origin = 'scale_factor',
            false_easting = 'false_easting',
            false_northing = 'false_northing',
            )

    _proj4_def = """+proj=stere
                +lon_0=straight_vertical_longitude_from_pole
                +lat_0=latitude_of_projection_origin
                +x_0=false_easting
                +y_0=false_northing
                +lat_ts=standard_parallel
                +k_0=scale_factor_at_projection_origin
            """

    def _check_params(self, kwargs):
        msg = "need to provide latitude_of_projection_origin (+lat_0) with -90 or +90"
        if not 'latitude_of_projection_origin' in kwargs \
                or kwargs['latitude_of_projection_origin'] not in (-90, 90):
            raise ValueError(msg)


class RotatedPole(CF_CRS, ccrs.RotatedPole):

    grid_mapping_name = "rotated_latitude_longitude"

    _cartopy = dict( 
            grid_north_pole_longitude ='pole_longitude', 
            grid_north_pole_latitude ='pole_latitude')

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

    @classmethod
    def _proj4_to_cf_params(cls, proj4_params):
        # can't just parse _proj4_def
        proj4_params = proj4_params.copy()
        import math
        assert proj4_params.pop('proj') == 'ob_tran'
        assert proj4_params.pop('o_lon_p',0) == 0.
        assert proj4_params.pop('o_proj','latlon') == 'latlon'
        assert abs(proj4_params.pop('to_meter',math.radians(1)) - math.radians(1)) < 1e-6

        cf = {}
        #cf['grid_north_pole_latitude'] = proj4_params.pop('o_lat_p', None)

        if 'o_lat_p' in proj4_params:
            cf['grid_north_pole_latitude'] = proj4_params.pop('o_lat_p')
        if 'lon_0' in proj4_params:
            cf['grid_north_pole_longitude'] = proj4_params.pop('lon_0') - 180.

        cf['grid_mapping_name'] = cls.grid_mapping_name

        if len(proj4_params) > 0:
            warnings.warn("some params left: {}".format(proj4_params))

        return cf

class TransverseMercator(CF_CRS, ccrs.TransverseMercator):
    grid_mapping_name = "transverse_mercator"

    _cartopy = dict(
            scale_factor_at_central_meridian = "scale_factor", 
            longitude_of_central_meridian = "central_longitude",
            latitude_of_projection_origin = "central_latitude",
            false_northing = "false_northing",
            false_easting = "false_easting",
            )

    _proj4_def = """+ellps=ellipsoid +proj=tmerc 
                    +lon_0=longitude_of_central_meridian 
                    +lat_0=latitude_of_projection_origin 
                    +k=scale_factor_at_central_meridian 
                    +x_0=false_easting
                    +y_0=false_northing
                    +units=IGNORE
                 """


#def _check_default(kwargs, nm, val):
#    assert not nm in kwargs or kwargs[nm] == val

# They are also based on inspection of this module's CF_CRS classes and
# in particular their attributes _proj4_def to match back pro4_params to CF params
#
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

def _init_from_proj4(proj4_params, strict=False):
    """ Find CF_CRS class corresponding to a dict of PROJ.4 parameters

    by inspecting the current module.

    Parameters
    ----------
    proj4_params : dict of PROJ.4 parameters
    strict : False by default, optional
        if True, only returns CF_CRS and raises and error otherwise
    
    Returns
    -------
    crs : dimarray.geo.crs.CF_CRS instance if possible
        or dimarray.geo.crs.Proj4 otherwise

    Raises
    ------
    ValueError if no CF_CRS subclass is found
    """
    if isinstance(proj4_params, str):
        proj4_params = dict(_parse_proj4(proj4_params))

    proj = proj4_params['proj'] # projection name

    # All available classes
    classes = _get_cf_crs_classes()

    # Get those whose proj_4 def matches
    patt = '+proj='+proj
    classes = filter(lambda cls: patt in cls._proj4_def,  classes)

    # Now try to initialize each one with proj4_params
    valid_crs = []
    for cls in classes:
        try:
            crs = cls.from_proj4(proj4_params)
            valid_crs.append(crs)
        except:
            pass

    # classes are already sorted by priority, so just return the first one
    try: 
        crs = valid_crs[0]
    except:
        msg = "{} matching (but failed) CRS classes for +proj={} ".format(len(classes), proj)
        msg += "initialize default Proj4 instance (no CF- equivalent)"
        print(msg)
        if strict : raise
        crs = Proj4(**proj4_params) 

    return crs


class Proj4(ccrs.CRS):
    """ cartopy.crs.CRS instance initialize from a PROJ.4 string
    """
    def __init__(self, proj4_init=None, **proj4_params):
        """ initialize a CRS instance based on PROJ.4 parameters

        Examples
        --------
        >>> prj = Proj4("+ellps=WGS84 +proj=stere +lat_0=90.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +lat_ts=71.0")
        """
        table_globe = dict([['datum', 'datum'], ['ellps', 'ellipse'],
                        ['a', 'semimajor_axis'], ['b', 'semiminor_axis'],
                        ['f', 'flattening'], ['rf', 'inverse_flattening'],
                        ['towgs84', 'towgs84'], ['nadgrids', 'nadgrids']])

    
        assert proj4_init is not None or len(proj4_params) > 0, "no argument provided"
        assert not (proj4_init is not None and len(proj4_params) > 0), "must provide EITHER a string of key-word arguments"
        if len(proj4_params) == 0:
            proj4_params = _parse_proj4(proj4_init) # key, value pair
        else:
            proj4_params = zip(proj4_params.keys(), proj4_params.values())

        # split parameters between globe and non-globe parameters
        globe_params = {table_globe[k]:v for k,v in proj4_params if k in table_globe.keys()}
        proj4_params = [(k,v) for k,v in proj4_params if k not in table_globe.keys()]

        # initialize Globe instance
        globe = ccrs.Globe(**globe_params)

        # initialize CRS instance
        super(Proj4, self).__init__(proj4_params, globe=globe)

    def to_cf(self, strict=True):
        """ transform to a CF_CRS instance
        """
        return _init_from_proj4(proj4_params, strict=strict)

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


def get_grid_mapping(grid_mapping, cf_conform=False):
    """ Get a cartopy's CRS instance from a variety of key-word parameters

    Parameters
    ----------
    grid_mapping : str or dict or cartopy.crs.CRS instance
        See below
    cf_conform : if True, only returns dimarray.geo.CF_CRS instances

    Returns
    -------
    cartopy's CRS instance 

    Note
    ----
    A grid mapping can be defined in one of the following ways:
    - providing a cartopy.crs.CRS instance directly
    - provide a cartopy.crs.CRS subclass name, for initialization with 
      default parameters
    - providing a dictionary of netCDF-conform parameters (CF-1.4)
    - provide a PROJ.4 string, with parameters preceded by '+' (EXPERIMENTAL)

    This will be converted into a cartopy.crs.CRS instance.

    Information on netcdf-conforming parameters can be found here:
    http://cfconventions.org/1.4.html#appendix-grid-mappings
    (CF-1.6 also exists, in track-change mode w.r.t CF-1.4)

    Information on PROJ.4 projections and parameters here:
    https://trac.osgeo.org/proj/wiki/GenParms
    with a list of transformations (and some explanations) there:
    http://www.remotesensing.org/geotiff/proj_list

    Examples
    --------

    Import Cartopy's crs module

    >>> import cartopy.crs as ccrs

    Longitude / Latitude coordinates

    >>> grid_mapping = ccrs.Geodetic() # cartopy
    >>> grid_mapping = "geodetic"  # cartopy class name
    >>> grid_mapping = {'grid_mapping_name':'longitude_latitude'} # CF-1.4

    Other coordinates systems onto which lon and lat can be projected onto

    >>> from dimarray.geo.crs import get_grid_mapping

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
    >>> crs = get_grid_mapping(grid_mapping)
    >>> crs.transform_point(70, -40, ccrs.Geodetic())
    (24969236.85758362, 8597597.732836112)

    ... PROJ.4 equivalent, with all parameters

    >>> proj4_init = "+ellps=WGS84 +proj=stere +lat_0=90.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +lat_ts=71.0"
    >>> crs = get_grid_mapping(proj4_init)
    >>> crs.transform_point(70, -40, ccrs.Geodetic())
    (24969236.85758362, 8597597.732836112)
    """
    if isinstance(grid_mapping, dict):
        grid_mapping = grid_mapping.copy()
        try:
            grid_mapping_name = grid_mapping.pop('grid_mapping_name')
        except KeyError:
            raise ValueError("grid_mapping_name not present")

        cls = _get_cf_crs_class(grid_mapping_name)
        grid_mapping = cls(**grid_mapping)

    elif isinstance(grid_mapping, str):

        # special case: PROJ.4 string?
        if '+' in grid_mapping:
            grid_mapping = _init_from_proj4(grid_mapping, strict=cf_conform)

        # common case: Cartopy class name
        else:
            members = inspect.getmembers(ccrs, lambda x: np.issubclass_(x, ccrs.CRS) and x.__name__.lower() == grid_mapping.lower())
            if len(members) == 0:
                raise ValueError("class "+grid_mapping+" not found")
            else:
                assert len(members) == 1, "more than one match, bug: check case"
                cls = members[0][1]
                #cls = getattr(ccrs, grid_mapping)
            grid_mapping = cls()

    elif not isinstance(grid_mapping, ccrs.CRS):
        #msg = 80*'-'+'\n'+get_grid_mapping.__doc__+'\n'+80*'-'+'\n\n'
        msg = 'grid_mapping: must be str or dict or cartopy.crs.CRS instance, got: {}'.format(grid_mapping)
        raise TypeError(msg)

    # just checking
    assert isinstance(grid_mapping, ccrs.CRS), 'something went wrong'

    if cf_conform and not isinstance(grid_mapping, CF_CRS):
        grid_mapping = _init_from_proj4(grid_mapping.proj4_params) # CF-conform grid-mapping

    return grid_mapping




### #
### # now do the actual projection
### # 
### @format_doc(grid_mapping=_doc_grid_mapping)
### def transform_coords(x, y, grid_mapping1, grid_mapping2):
###     """ Transform coordinates pair into another coordinate system
### 
###     This is a wrapper around cartopy.crs.CRS.transform_point(s)
### 
###     Parameters
###     ----------
###     x, y : coordinates in grid_mapping1 (scalar or array-like)
###     grid_mapping1 : coordinate system of input x and y (str or dict or cartopy.crs.CRS instance)
###     grid_mapping2 : target coordinate system (str or dict or cartopy.crs.CRS instance)
### 
###     Returns
###     -------
###     xt, yt : (transformed) coordinates in grid_mapping2 
### 
###     Note
###     ----
###     {grid_mapping}
### 
###     See Also
###     --------
###     dimarray.geo.GeoArray.transform
###     dimarray.geo.GeoArray.transform_vectors
###     """
###     grid_mapping1 = get_grid_mapping(grid_mapping1)
###     grid_mapping2 = get_grid_mapping(grid_mapping2)
### 
###     if np.isscalar(x):
###         xt, yt = grid_mapping2.transform_point(x, y, grid_mapping1)
###     else:
###         xt, yt = grid_mapping2.transform_points(grid_mapping1, x, y)
### 
###     return xt, yt
### 
### 
### def transform_vectors(x, y, u, v, grid_mapping1, grid_mapping2):
###     """ Transform vectors from one coordinate system to another
### 
###     This is a wrapper around cartopy.crs.CRS.transform_vectors
### 
###     Parameter
###     ---------
###     x, y : source coordinates in grid_mapping1
###     u, v : source vector components in grid_mapping1
###     grid_mapping1, grid_mapping2 : source and destination grid mappings
### 
###     Returns
###     -------
###     ut, vt : transformed vector components 
### 
###     Note
###     ----
###     Need to transform coordinates separately
### 
###     See Also
###     --------
###     geo.transform_coords
###     """
###     grid_mapping1 = get_grid_mapping(grid_mapping1)
###     grid_mapping2 = get_grid_mapping(grid_mapping2)
### 
###     ut, vt = grid_mapping2.transform_vectors(grid_mapping1, x, y, u, v)
### 
###     return ut, vt
### 
### 
### def project_coords(lon, lat, grid_mapping, inverse=False):
###     """ Project lon / lat onto a grid mapping
### 
###     This is a wrapper around cartopy.crs.CRS.transform_point(s)
### 
###     Parameters
###     ----------
###     lon, lat : longitude and latitude (if not inverse)
###     grid_mapping : dict of netcdf-conforming grid mapping specification, or cartopy's CRS instance
###     inverse : bool, optional
###         if True, do the reverse transformation from x, y into lon, lat
### 
###     Return
###     ------
###     x, y : coordinates on the projection plane
### 
###     See Also
###     --------
###     dimarray.geo.transform_coords
###     """
###     if inverse:
###         x, y = transform_coords(lon, lat, grid_mapping, ccrs.Geodetic())
###     else:
###         x, y = transform_coords(lon, lat, ccrs.Geodetic(), grid_mapping)
### 
###     return x, y
