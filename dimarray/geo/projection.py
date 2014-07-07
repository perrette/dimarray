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
NetCDF Conventions:
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
from cartopy import crs
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
        crs.Stereographic = compat_crs.Stereographic
        crs.NorthPolarStereo = compat_crs.NorthPolarStereo
        crs.SouthPolarStereo = compat_crs.SouthPolarStereo

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

#class Globe(crs.Globe):
#    """ Represents the globe
#    """
#    # netCDF to Cartopy parameters
#    _table = dict(datum='datum', ellipsoid='ellipse', 
#            semi_major_axis='semimajor_axis', 
#            semi_minor_axis='semiminor_axis', 
#            flattening='flattening', inverse_flattening='inverse_flattening', 
#            towgs84='towgs84', nadgrids='nadgrids')
#
#    def __init__(self, **nc_params):
#        """
#        Same as Cartopy's Globe class but with parameter names following netCDF conventions
#        Equivalent with Cartopy's parameter projection can be found in 
#        dimarray.geo.projection.Globe._table
#        """
#        cartopy_params = {}
#        for k in nc_params.keys():
#            try:
#                cartopy_params[self._table[k]] = nc_params[k]
#            except KeyError:
#                msg = "{} unexpected. Accepted parameters: {}".format(k, self._table.keys())
#                raise ValueError(msg)
#
#        try:
#            crs.Globe.__init__(self, **cartopy_params)
#        except Exception as error:
#            msg = _error_message(error, self)
#            raise error.__class__(msg)


class Projection(object):
    """ Projection class to map Cartopy's implementation onto netCDF parameter names
        
    To be subclassed
    """
    # Equivalent of globe parameters between CF-1.4 (keys) and cartopy (values)
    _table_globe = dict(datum='datum', ellipsoid='ellipse', 
            semi_major_axis='semimajor_axis', 
            semi_minor_axis='semiminor_axis', 
            flattening='flattening', inverse_flattening='inverse_flattening', 
            towgs84='towgs84', nadgrids='nadgrids')

    _table = {} # correspondence between netCDF and Cartopy names
    _alternate_parameters = [] # list of list of alternate parameters that are redundant
    grid_mapping_name = None

    def __init__(self, **kwargs):
        """ Initialize a projection class with CF-1.4 conforming parameters

        See class's `_table` and `_table_globe` attribute for correspondence 
        between CF-1.4 and Cartopy parameters
        """
        # check for redundant parameters
        for params in self._alternate_parameters:
            _assert_alternate_params(kwargs, params)

        cartopy_params = {}
        globe_params = {} # also Cartopy format
        #globe_kwargs = {}
        for k in kwargs.keys(): # CF-1.4 format
            if k in self._table.keys():
                cartopy_params[self._table[k]] = kwargs[k]
            elif k in self._table_globe.keys():
                globe_params[self._table_globe[k]] = kwargs[k]
            else:
                msg = "{} unexpected. Accepted parameters: {}".format(k, self._table.keys()+self._table_globe.keys())
                raise ValueError(msg)

        # Initialize a globe
        try:
            globe = crs.Globe(**globe_params)
            cartopy_params['globe'] = globe
        except Exception as error:
            msg = _error_message(error, self._table_globe)
            raise error.__class__(msg)

        # Initialize cartopy coordinate system
        try:
            CRSproj = self.__class__.__bases__[1] # (Projection, crs...)
            assert np.issubclass_(CRSproj, crs.Projection) # must be a crs projection
            CRSproj.__init__(self, **cartopy_params)

        except Exception as error:
            msg = _error_message(error, self._table)
            raise error.__class__(msg)

#
# NOTE: the grid_mapping_name is 'longitude_latitude' and not 'geodetic'
#
class Geodetic(Projection, crs.Geodetic):
    """ Same as cartopy.crs.Geodetic but accept netCDF parameters as arguments
    """
    grid_mapping_name = 'longitude_latitude'
    _table = {} # no parameter apart from "globe"

class Stereographic(Projection, crs.Stereographic):

    grid_mapping_name = "stereographic"

    # correspondence table between netCDF and Cartopy parameters
    _table = dict(
            longitude_of_projection_origin = 'central_longitude',
            latitude_of_projection_origin = 'central_latitude',
            false_easting = 'false_easting',
            false_northing = 'false_northing',
            scale_factor_at_projection_origin = 'NO DIRECT EQUIVALENT IN CARTOPY, transformed into standard_parallel somehow',
            standard_parallel = 'true_scale_latitude', # this is not the preferred way according to CF-1.4
            )

    def __init__(self, **kwargs):

        # transform standard_parallel to scale_factor_at_projection_origin
        _assert_alternate_params(kwargs, ['standard_parallel', 'scale_factor_at_projection_origin'])
        if 'standard_parallel':
            warnings.warn('Use "scale_factor_at_projection_origin" instead of "standard_parallel" in Stereographic projection')
#
#
#
class PolarStereographic(Projection, crs.Stereographic):

    grid_mapping_name = "polar_stereographic"

    # correspondence table between netCDF and Cartopy parameters
    _table = dict(
            straight_vertical_longitude_from_pole = 'central_longitude',  # preferred?
            longitude_of_projection_origin = 'central_longitude', # also accepted
            latitude_of_projection_origin = 'central_latitude',
            standard_parallel = 'true_scale_latitude',
            scale_factor_at_projection_origin = 'alternate formulation to standard_parallel, equal to (1+abs(sin(radians(stdpar))))/2',
            false_easting = 'false_easting',
            false_northing = 'false_northing',
            )

    def __init__(self, **kwargs):

        # transform standard_parallel to scale_factor_at_projection_origin
        _assert_alternate_params(kwargs, ['standard_parallel', 'scale_factor_at_projection_origin'])
        _assert_alternate_params(kwargs, ['straight_vertical_longitude_from_pole', 'longitude_of_projection_origin'])

        if 'scale_factor_at_projection_origin' in kwargs:
            scale = kwargs.pop('scale_factor_at_projection_origin')
            sin = (scale * 2 - 1)
            stdpar = math.degrees(math.asin(sin))
            kwargs['standard_parallel'] = stdpar

        super(PolarStereographic, self).__init__(**kwargs)

    __init__.__doc__ = Projection.__doc__ # update doc


## #
## # Here just for convenience
## #
## class NorthPolarStereographic(Projection, crs.NorthPolarStereo):
## 
##     grid_mapping_name = "north_polar_stereographic" # not in CF-convention
## 
##     # correspondence table between netCDF and Cartopy parameters
##     _table = dict(
##             straight_vertical_longitude_from_pole = 'central_longitude',  # preferred?
##             longitude_of_projection_origin = 'central_longitude', # also accepted
##             )
## 
##     _alternate_parameters = [['straight_vertical_longitude_from_pole', 'longitude_of_projection_origin']]
## 
## class SouthPolarStereographic(Projection, crs.SouthPolarStereo):
## 
##     grid_mapping_name = "south_polar_stereographic" # not in CF-convention
## 
##     # correspondence table between netCDF and Cartopy parameters
##     _table = dict(
##             straight_vertical_longitude_from_pole = 'central_longitude',  # preferred?
##             longitude_of_projection_origin = 'central_longitude', # also accepted
##             )
## 
##     _alternate_parameters = [['straight_vertical_longitude_from_pole', 'longitude_of_projection_origin']]


#
# Get one of the classes defined above by name
#
def _get_grid_mapping_class(grid_mapping_name):
    """ return a grid mapping class which accepts 
    """
    # special case: PROJ.4 arguments are already provided

    current = sys.modules[__name__] # current module
    found = inspect.getmembers(current , lambda x: np.issubclass_(x, Projection) and x.grid_mapping_name == grid_mapping_name)

    if len(found) == 0:
        found_all = inspect.getmembers(current , lambda x: np.issubclass_(x, Projection) and x.grid_mapping_name is not None)
        accepted = [tup[1].grid_mapping_name for tup in found_all]

        msg = "Unknown projection: {}. Currently accepted projections: {}".format(grid_mapping_name, accepted)
        msg += "\n"+file_an_issue_message()
        raise ValueError(msg)

    assert len(found) == 1, "several matches !"

    return found[0][1] # 0: class name, 1: class

# Get a Projection instance. 
# See doc string below
_doc_grid_mapping = """
    A grid mapping can be defined in one of the following ways:
    - providing a cartopy.crs.CRS instance directly
    - provide a cartopy.crs.CRS subclass name, for initialization with 
      default parameters
    - providing a dictionary of netCDF-conform parameters (CF-1.4)
    - provide a PROJ.4 string, with parameters preceded by '+' (EXPERIMENTAL)

    Information on netcdf-conforming parameters can be found here:
    http://cfconventions.org/1.4.html#appendix-grid-mappings
    (CF-1.6 also exists, in track-change mode w.r.t CF-1.4)

    Information on PROJ.4 projections and parameters here:
    https://trac.osgeo.org/proj/wiki/GenParms
    with a list of transformations (and some explanations) there:
    http://www.remotesensing.org/geotiff/proj_list

    Examples
    --------

    All of the grid_mapping instances below could be checked by
    calling dimarray.geo.projection._check_grid_mapping
    The returned cartopy.crs.CRS instances all have a proj4_init 
    attribute for PROJ.4 equivalent.

    Import Cartopy's crs module

    >>> import cartopy.crs as ccrs
    
    Longitude / Latitude coordinates

    >>> grid_mapping = ccrs.Geodetic() # cartopy
    >>> grid_mapping = "geodetic"  # cartopy class name
    >>> grid_mapping = {'grid_mapping_name':'longitude_latitude'} # CF-1.4

    North Polar Stereographic Projection over Greenland

    ... with Cartopy Stereographic class, and all parameters
        (note true_scale_latitude only makes sense for polar stereo)

    >>> globe = ccrs.Globe(ellipse='WGS84')  # it's the default anyway
    >>> grid_mapping = ccrs.Stereographic(  
    ...     central_latitude  = 90.,  # center of projection 
    ...     central_longitude = -39., # center of projection
    ...     true_scale_latitude = 71., # only makes sense for polar stereo
    ...     false_easting = 0.,  # default offset to express x w.r.t 
    ...     false_northing = 0., # default offset to express y
    ...     globe=globe)  # cartopy.crs.CRS instance

    ... with Cartopy NorthPolarStereo class (central_latitude=90. by default)

    >>> grid_mapping = ccrs.NorthPolarStereo(central_longitude=-39., true_scale_latitude=71.)  

    ... same as above, default parameters (central_longitude=0., 
                                    true_scale_latitude=None, or 90.)

    >>> grid_mapping = 'northpolarstereo'  # cartopy instance with defualt params 

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

    ... PROJ.4 equivalent, with all parameters

    >>> from dimarray.geo.projection import _check_grid_mapping
    >>> gm = _check_grid_mapping(grid_mapping)
    >>> isinstance(gm, ccrs.CRS)
    True
    >>> gm.proj4_init
    '+ellps=WGS84 +proj=stere +lat_0=90.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +lat_ts=71.0 +no_defs'
""".strip()

def _check_grid_mapping(grid_mapping):
    """ check grid_mapping type (see doc in transform_coords)
    """
    if isinstance(grid_mapping, dict):
        cls = _get_grid_mapping_class(grid_mapping.pop('grid_mapping_name'))
        grid_mapping = cls(**grid_mapping)

    elif isinstance(grid_mapping, str):

        # special case: PROJ.4 string?
        if '+' in grid_mapping:
            grid_mapping = _get_from_proj4_args(grid_mapping)

        # common case: netCDF transformation name: default parameters
        else:
            cls = _get_grid_mapping_class(grid_mapping)
            grid_mapping = cls()

    elif not isinstance(grid_mapping, crs.CRS):
        msg = 80*'-'+'\n'+_doc_grid_mapping+'\n'+80*'-'+'\n\n'
        msg += 'grid_mapping: must be str or dict or cartopy.crs.CRS instance (see above)'
        raise TypeError(msg)

    return grid_mapping


#
# now do the actual projection
# 


@format_doc(grid_mapping=_doc_grid_mapping)
def transform_coords(x, y, grid_mapping1, grid_mapping2):
    """ Transform coordinates pair into another coordinate system

    This is a wrapper around cartopy.crs.CRS.transform_point(s)

    Parameters
    ----------
    x, y : coordinates in grid_mapping1 (scalar or array-like)
    grid_mapping1 : coordinate system of input x and y (str or dict or cartopy.crs.CRS instance)
    grid_mapping2 : target coordinate system (str or dict or cartopy.crs.CRS instance)

    Returns
    -------
    xt, yt : (transformed) coordinates in grid_mapping2 

    Note
    ----
    {grid_mapping}

    See Also
    --------
    dimarray.geo.GeoArray.transform
    dimarray.geo.GeoArray.transform_vectors
    """
    grid_mapping1 = _check_grid_mapping(grid_mapping1)
    grid_mapping2 = _check_grid_mapping(grid_mapping2)

    if np.isscalar(x):
        xt, yt = grid_mapping2.transform_point(x, y, grid_mapping1)
    else:
        xt, yt = grid_mapping2.transform_points(grid_mapping1, x, y)

    return xt, yt


def transform_vectors(x, y, u, v, grid_mapping1, grid_mapping2):
    """ Transform vectors from one coordinate system to another

    This is a wrapper around cartopy.crs.CRS.transform_vectors

    Parameter
    ---------
    x, y : source coordinates in grid_mapping1
    u, v : source vector components in grid_mapping1
    grid_mapping1, grid_mapping2 : source and destination grid mappings

    Returns
    -------
    ut, vt : transformed vector components 

    Note
    ----
    Need to transform coordinates separately

    See Also
    --------
    geo.transform_coords
    """
    grid_mapping1 = _check_grid_mapping(grid_mapping1)
    grid_mapping2 = _check_grid_mapping(grid_mapping2)

    ut, vt = grid_mapping2.transform_points(grid_mapping1, x, y, u, v)

    return ut, vt


def project_coords(lon, lat, grid_mapping, inverse=False):
    """ Project lon / lat onto a grid mapping

    This is a wrapper around cartopy.crs.CRS.transform_point(s)

    Parameters
    ----------
    lon, lat : longitude and latitude (if not inverse)
    grid_mapping : dict of netcdf-conforming grid mapping specification, or Projection instance
    inverse : bool, optional
        if True, do the reverse transformation from x, y into lon, lat

    Return
    ------
    x, y : coordinates on the projection plane

    See Also
    --------
    dimarray.geo.transform_coords
    """
    if inverse:
        x, y = transform_coords(lon, lat, grid_mapping, crs.Geodetic())
    else:
        x, y = transform_coords(lon, lat, crs.Geodetic(), grid_mapping)

    return x, y
