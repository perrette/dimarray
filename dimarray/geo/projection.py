""" Perform coordinatates transformations based on cartopy, 
but using standard netCDF names as arguments

The aim is to be able to read and transform coordinate systems from a netCDF looking like:
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
"""
import warnings
import math
import numpy
import cartopy
from cartopy import crs
from dimarray.info import file_an_issue_message
from projection_def_netcdf import get_best_matches_param, get_best_matches_mapping
#from shapely.geometry import Point, LineString

# check cartopy version
M = cartopy.__version__.split('.')[0]
m = cartopy.__version__.split('.')[1]
if int(M) == 0 and int(m) < 11:
    warnings.warn('Projections were only tested for cartopy versions 0.11.x')

def get_grid_mapping_class(grid_mapping_name):

    if grid_mapping_name in ('geodetic','longlat'):
        return crs.Geodetic

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

def _error_message(error):
    " make a typical error message when initializing a Cartopy projection fails"
    msg = "Cartopy Exception:\n\n"
    msg += error.message
    msg += "\n\n----------------------------------------------------"
    msg += "\n"
    msg += "\nProblem when initializing Cartopy's projection class"
    msg += "\nSee corresponding Cartopy documentation for more info"
    msg += "\nThe table of correspondence between netCDF and Cartopy"
    msg += "\nparameters is {}.".format(self._table)
    msg += "\n"
    msg += "\n"+file_an_issue_message()
    return msg


class Globe(crs.Globe):
    """ Represents the globe
    """
    # netCDF to Cartopy parameters
    _table = dict(datum='datum', ellipsoid='ellipse', 
            semi_major_axis='semimajor_axis', 
            semi_minor_axis='semiminor_axis', 
            flattening='flattening', inverse_flattening='inverse_flattening', 
            towgs84='towgs84', nadgrids='nadgrids')

    def __init__(self, **nc_params):
        """
        Same as Cartopy's Globe class but with parameter names following netCDF conventions
        Equivalent with Cartopy's parameter projection can be found in 

        Parameters
        ----------

        datum - Proj4 "datum" definiton. Default to no datum.
        ellipsoid - Proj4 "ellps" definiton. Default to 'WGS84'.
        semi_major_axis - Semimajor axis of the spheroid / ellipsoid.
        semi_minor_axis - Semiminor axis of the ellipsoid.
        flattening - Flattening of the ellipsoid.
        inverse_flattening - Inverse flattening of the ellipsoid.
        towgs84 - Passed through to the Proj4 definition.
        nadgrids - Passed through to the Proj4 definition.
        """
        cartopy_params = {}
        for k in nc_params.keys():
            try:
                cartopy_params[self._table[k]] = nc_params[k]
            except KeyError:
                msg = "{} unexpected. Accepted parameters: {}".format(k, self._table.keys())
                raise ValueError(msg)

        try:
            crs.Globe.__init__(self, **cartopy_params)
        except Exception as error:
            msg = _error_message(error)
            raise error.__class__(msg)


def assert_alternate_params(kwargs, params):
    """ check for redundant parameters
    """
    matching = []
    for k in kwargs:
        if k in params:
            matching.append(k)

    if len(matching) > 1:
        raise ValueError("Redundant parameter definition, please use only one of: "+", ".join(matching))


class Projection(object):
    """ Projection class to map Cartopy's implementation onto netCDF parameter names
        
    To be subclassed
    """
    _table = {} # correspondence between netCDF and Cartopy names
    _alternate_parameters = [] # list of list of alternate parameters that are redundant
    grid_mapping_name = None

    def __init__(self, **nc_params):
        """ Initialize a projection class with netCDF parameters

        See class's `_table` attribute for correspondence between netCDF and Cartopy parameters

        CF-conform parameters were taken from:
        https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/reference/StandardCoordinateTransforms.html

        See help on Cartopy's projection, for example for the Stereographic projection:
        >>> from cartopy import crs
        >>> help(crs.Stereographic) # doctest: +SKIP
        """
        # check for redundant parameters
        for params in self._alternate_parameters:
            assert_alternate_params(kwargs, params)

        cartopy_params = {}
        globe_params = {}
        for k in nc_params.keys():
            if k in self._table.keys():
                cartopy_params[self._table[k]] = nc_params[k]
            elif k in Globe._table.keys():
                globe_params[Globle._table[k]] = nc_params[k]
            else:
                msg = "{} unexpected. Accepted parameters: {}".format(k, self._table.keys()+Globle._table.keys())
                raise ValueError(msg)

        globe = Globe(**globe_params)
        cartopy_params['globe'] = globe

        # get equivalent CRS projections
        try:

            ## get the CRS projection
            #for base in self.__class__.__bases__:
            #    if np.issubclass_(base, crs.Projection):
            #        CRSproj = base
            #        break
            CRSproj = self.__class__.__bases__[1]  # first base is Projection, second is 
            assert np.issubclass_(CRSproj, crs.Projection) # must be a crs projection
            CRSproj.__init__(self, **cartopy_params)

        except Exception as error:
            msg = "Cartopy Exception:\n\n"
            msg += error.message
            msg += "\n\n----------------------------------------------------"
            msg += "\n"
            msg += "\nProblem when initializing Cartopy's projection class"
            msg += "\nSee corresponding Cartopy documentation for more info"
            msg += "\nThe table of correspondence between netCDF and Cartopy"
            msg += "\nparameters is {}.".format(self._table)
            msg += "\n"
            msg += "\n"+file_an_issue_message()
            raise error.__class__(msg)

class Stereographic(Projection, crs.Stereographic):

    grid_mapping_name = "stereographic"

    # correspondence table between netCDF and Cartopy parameters
    _table = dict(
            longitude_of_projection_origin = 'central_longitude',
            latitude_of_projection_origin = 'central_latitude',
            false_easting = 'false_easting',
            false_northing = 'false_northing',
            scale_factor_at_projection_origin = 'true_scale_latitude',
            )

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
            standard_parallel = 'alternate formulation to true_scale_latitude: arcsin( true_scale_latitude * 2 - 1 ), in degrees', 
            scale_factor_at_projection_origin = 'true_scale_latitude',
            false_easting = 'false_easting',
            false_northing = 'false_northing',
            )

    def __init__(self, **kwargs):

        # transform standard_parallel to scale_factor_at_projection_origin
        assert_alternate_params(kwargs, ['standard_parallel', 'scale_factor_at_projection_origin'])
        assert_alternate_params(kwargs, ['straight_vertical_longitude_from_pole', 'longitude_of_projection_origin'])

        if 'standard_parallel' in kwargs:
            stdpar = kwargs.pop('standard_parallel')
            sin = abs(math.sin( math.radians( stdpar)))
            scale = (1+sin)/2
            kwargs['scale_factor_at_projection_origin'] = scale

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


##
## now do the actual projection
## 
def project_coord(lon, lat, grid_mapping, inverse=False):
    """ Project lon / lat onto a grid mapping

    Parameters
    ----------
    lon, lat : longitude and latitude (if not inverse)
    grid_mapping : dict of netcdf-conforming grid mapping specification, or Projection instance
    inverse : bool, optional
        if True, do the reverse transformation from x, y into lon, lat

    Return
    ------
    x, y : coordinates on the projection plane

    Source
    ------
    https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/reference/StandardCoordinateTransforms.html

    See Also
    --------
    dimarray.geo.transform
    """
    if inverse:
        x, y = transform(lon, lat, grid_mapping, crs.Geodetic)
    else:
        x, y = transform(lon, lat, crs.Geodetic, grid_mapping)

    return x, y


def transform_coord(x, y, grid_mapping1, grid_mapping2):
    """ Transform coordinate pair into another coordinate

    Parameters
    ----------
    x, y : coordinates in grid_mapping1 (scalar or array-like)
    grid_mapping1 : coordinate system of input x and y (str or dict or cartopy.crs.CRS instance)
    grid_mapping2 : target coordinate system (str or dict or cartopy.crs.CRS instance)

    Returns
    -------
    x2, y2 : coordinates in grid_mapping2 

    Note
    ----
    Grid mapping specification can be dict of netcdf-conforming parameters 
    or cartopy.crs.CRS instance

    In particular, cartopy.crs.Geodetic defines a longitude / latitude projection system

    Grid-mapping can also be a string representing the grid_mapping_name, 
    in which case default default parameter values (as defined by original 
    cartopy classes) are used.  
    
    "geodetic" or "longlat" will define a cartopy.crs.Geodetic instance

    Information on netcdf-conforming parameters can be found here:
    https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/reference/StandardCoordinateTransforms.html

    See Also
    --------
    dimarray.geo.project
    """
    if isinstance(grid_mapping1, dict):
        cls1 = get_grid_mapping_class(grid_mapping1.pop('grid_mapping_name'))
        grid_mapping1 = cls1(**grid_mapping1)
    elif isinstance(grid_mapping1, str):
        cls1 = get_grid_mapping_class(grid_mapping1)
        grid_mapping1 = cls1()
    elif not isinstance(grid_mapping1, crs.CRS):
        raise TypeError('grid_mapping1: must be str or dict or cartopy.crs.CRS instance')

    if isinstance(grid_mapping2, dict):
        cls2 = get_grid_mapping_class(grid_mapping2.pop('grid_mapping_name'))
        grid_mapping2 = cls2(**grid_mapping2)
    elif isinstance(grid_mapping2, str):
        cls2 = get_grid_mapping_class(grid_mapping2)
        grid_mapping2 = cls2()
    elif not isinstance(grid_mapping2, crs.CRS):
        raise TypeError('grid_mapping2: must be str or dict or cartopy.crs.CRS instance')

    if np.isscalar(x):
        x, y = grid_mapping2.transform_point(lon, lat, grid_mapping1)
    else:
        x, y = grid_mapping2.transform_points(grid_mapping1, lon, lat)

    return x, y
