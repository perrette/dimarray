""" Geo-Array transformation between various coordinate reference system
""" 
import warnings
import numpy as np
from dimarray.geo.geoarray import GeoArray, DimArray, Coordinate, Axis
from dimarray.geo.geoarray import X, Y, Z, Longitude, Latitude 
from dimarray import stack
from dimarray.compat.basemap import interp

#
# share private functions
#
def _check_horizontal_coordinates(geo_array, horizontal_coordinates=None, add_grid_mapping=True):
    """ return horizontal coordinates
    """
    if horizontal_coordinates: 
        assert len(horizontal_coordinates) == 2, "horizontal_coordinates must be a sequence of length 2"
        x0nm, y0nm = horizontal_coordinates
        x0 = geo_array.axes[x0nm]
        y0 = geo_array.axes[y0nm]

    else:
        xs = filter(lambda x: isinstance(x, X), geo_array.axes)
        ys = filter(lambda x: isinstance(x, Y), geo_array.axes)
        longs = filter(lambda x: isinstance(x, Longitude), geo_array.axes)
        lats = filter(lambda x: isinstance(x, Latitude), geo_array.axes)

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

    # Add grid mapping to GeoArray if Coordinates are Geodetic (long/lat)
    if add_grid_mapping and (not hasattr(geo_array, 'grid_mapping') or geo_array.grid_mapping is None) \
        and isinstance(x0, Longitude) and isinstance(y0, Latitude):
            geo_array.grid_mapping = {'grid_mapping_name':'latitude_longitude'}

    return x0, y0


def _check_grid_mapping(grid_mapping=None, geo_array=None):
    """ make sure the grid mapping is a CRS instance, 
    or convert it to a CRS instance
    """
    import cartopy.crs as ccrs
    from dimarray.geo.crs import get_grid_mapping, CF_CRS

    if grid_mapping is None and geo_array is not None \
        and hasattr(geo_array, 'grid_mapping'):
            grid_mapping = geo_array.grid_mapping

    if grid_mapping is None: 
        raise ValueError("grid mapping not provided")

    ## already a CRS instance : do nothing
    #if isinstance(grid_mapping, ccrs.CRS):
    #    return grid_mapping

    # if dict remove name
    if isinstance(grid_mapping, dict) and 'name' in grid_mapping.keys():
        del grid_mapping['name'] # typical case via ._metadata after reading from a dataset

    grid_mapping = get_grid_mapping(grid_mapping)

    # now add a cf_params attribute if needed
    # NOTE: for now just do not force conversion from Cartopy
    # since the CF version is still experimental
    if not isinstance(grid_mapping, CF_CRS):
        try:
            crs = get_grid_mapping(grid_mapping, cf_conform=True)
            grid_mapping.cf_params = grid_mapping
        except:
            raise

    return grid_mapping


def _add_grid_mapping_metadata(a, grid_mapping):
    """ add grid_mapping metadata to an array
    """
    try:
        cf_params = grid_mapping.cf_params
        a.grid_mapping = cf_params
    except:
        warnings.warn("cannot convert grid mapping to CF-conform metadata")
        if hasattr(a, 'grid_mapping'): del a.grid_mapping # remove old metadata

def _make_projection_coordinate(xt, yt, grid_mapping):
    """ create new projection coordinates based on grid mapping

    xt, yt : array-like
    grid_mapping : cartopy.crs.CRS instance (and better CF_CRS instance)
    """
    def _is_longlat(grid_mapping):
        import cartopy.crs as ccrs
        return isinstance(grid_mapping, ccrs.Geodetic) \
                or '+proj=latlong' in grid_mapping.proj4_init \
                or '+proj=lonlat' in grid_mapping.proj4_init \
                or '+proj=longlat' in grid_mapping.proj4_init \
                or '+proj=latlon' in grid_mapping.proj4_init 

    # ...toward geodetic coordinates? (several standards seem to exist, cartopy uses "lonlat"
    if _is_longlat(grid_mapping):

        xt = Longitude(xt)
        yt = Latitude(yt)

    # ...check whther the projection comes along with metadata
    else:

        # Make actual Coordinate axes
        if hasattr(grid_mapping, '_x_metadata'):
            meta = grid_mapping._x_metadata
            name = meta.pop('name', 'x')
            xt = X(xt, name)
            xt._metadata = meta

        else:
            xt = X(xt, 'x')

        # Make actual Coordinate axes
        if hasattr(grid_mapping, '_y_metadata'):
            meta = grid_mapping._x_metadata
            name = meta.pop('name', 'y')
            yt = Y(xt, name)
            yt._metadata = meta

        else:
            yt = Y(yt, 'y')

    return xt, yt

#
# Transformations involving changes in Coordinate Reference System
#
def transform_coords(from_grid_mapping, to_grid_mapping, x0, y0, xt=None, yt=None):
    """ project coordinates

    Parameters
    ----------
    from_grid_mapping, to_grid_mapping 
    x0, y0 : horiztonal coordinates in from_grid_mapping
    xt, yt : Axis instances or ndarrays, optional
        transformed axes to interpolate on
        (otherwise they will be determined) 

    Return
    ------
    xt, yt : new Axis instances
    x0_interp, y0_interp : coordinate in original grid mapping that yield a regular transformed grid
    """
    # Transform coordinates 
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    assert x0.ndim ==  1 and y0.ndim == 1
    x0_2d, y0_2d = np.meshgrid(x0, y0) # make 2-D coordinates
    trans = to_grid_mapping.transform_points(from_grid_mapping, x0_2d, x0_2d)
    xt_2d, yt_2d, zt_2d = np.rollaxis(trans,-1) # last dimension first

    # Make regular grid in new coordinate system for interpolation
    if xt is None:
        xt = np.linspace(xt_2d.min(), xt_2d.max(), x0.size) # keep about the same size
    if yt is None:
        yt = np.linspace(yt_2d.min(), yt_2d.max(), y0.size) # keep about the same size

    assert xt.ndim == 1 and yt.ndim == 1, 'transformed coordinates must be 1-D'

    # make it 2D
    xt_2dr, yt_2dr = np.meshgrid(xt, yt) # regular 2-D grid

    # transform regular grid back to the original CRS for interpolation 
    # (need regular grid as first argument)
    tmp = from_grid_mapping.transform_points(to_grid_mapping, xt_2dr, yt_2dr)
    x0_int, y0_int, z0_int = np.rollaxis(tmp, -1) # roll last axis first

    # Create new axes with appropriate metadata (unless alread Axis instances are provided)
    if isinstance(xt, Axis) and isinstance(yt, Axis):
        return 

    xt, yt =_make_projection_coordinate(xt, yt, to_grid_mapping)

    return xt, yt, x0_int, y0_int

#
#
#
def transform(geo_array, to_grid_mapping, from_grid_mapping=None, \
        xt=None, yt=None, horizontal_coordinates=None):
    """ Transform scalar field array into a new coordinate system and \
            interpolate values onto a new regular grid

    Parameters
    ----------
    geo_array : GeoArray or other DimArray instance
    to_grid_mapping : str or dict or cartopy.crs.CRS instance
        grid mapping onto which the transformation should be done
        str : PROJ.4 str or cartopy.crs.CRS class name
        dict : CF parameters
    from_grid_mapping : idem, optional
        original grid mapping, to be provided only when no grid_mapping 
        attribute is defined or when the axes are something else than 
        Longitude and Latitude
    xt, yt : array-like (1-D), optional
        new coordinates to interpolate the array on
        will be deduced as min and max of new coordinates if not provided
    horizontal_coordinates : sequence with 2 str, optional
        provide horizontal coordinates by name if not an instance of 
        Latitude, Longitude, X or Y

    Return
    ------
    transformed : GeoArray
        new GeoArray transformed

    Examples
    --------
    """ 
    # local import since it's quite heavy
    from dimarray.geo.crs import get_grid_mapping  

    if not isinstance(geo_array, DimArray):
        raise TypeError("geo_array must be a DimArray instance")
    if not isinstance(geo_array, GeoArray):
        geo_array = GeoArray(geo_array) 

    # find horizontal coordinates
    x0, y0 = _check_horizontal_coordinates(geo_array, horizontal_coordinates)

    # transpose the array to shape .., y0, x0 (cartesian convention needed for meshgrid)
    dims_orig = geo_array.dims
    dims_new = [d for d in geo_array.dims if d not in [x0.name, y0.name]] + [y0.name, x0.name]
    geo_array = geo_array.transpose(dims_new) 

    #assert geo_array.dims.index(x0.name) > geo_array.axes[

    # get cartopy.crs.CRS instances
    from_grid_mapping = _check_grid_mapping(from_grid_mapping, geo_array)
    to_grid_mapping = _check_grid_mapping(to_grid_mapping)
    #from_grid_mapping = _get_grid_mapping(geoarray, from_grid_mapping)
    #to_grid_mapping = get_grid_mapping(to_grid_mapping)

    # Transform coordinates and prepare regular grid for interpolation
    xt, yt, x0_interp, y0_interp = transform_coords(from_grid_mapping, to_grid_mapping, x0, y0, xt, yt)

    if geo_array.ndim == 2:
        #newvalues = interp(geo_array.values, xt_2d, yt_2d, xt_2dr, yt_2dr)
        newvalues = interp(geo_array.values, x0.values, y0.values, x0_interp, y0_interp)
        transformed = geo_array._constructor(newvalues, [yt, xt])

    else:
        # first reshape to 3-D, flattening everything except vertical coordinates
        # TODO: optimize by computing and re-using weights?
        obj = geo_array.group((x0.name, xt.name), reverse=True, insert=0)  
        newvalues = []
        for k, suba in obj.iter(axis=0): # iterate over the first dimension
            #newval = interp(suba.values, xt_2d, yt_2d, xt_2dr, yt_2dr)
            newval = interp(geo_array.values, x0.values, y0.values, x0_interp, y0_interp)
            newvalues.append(newval)

        # stack the arrays together
        newvalues = np.array(newvalues)
        grouped_obj = geo_array._constructor(newvalues, [obj.axes[0], yt, xt])
        transformed = grouped_obj.ungroup(axis=0)

    # reshape back
    # ...replace old axis names by new ones of the projection
    dims_orig = list(dims_orig)
    dims_orig[dims_orig.index(x0.name)] = xt.name
    dims_orig[dims_orig.index(y0.name)] = yt.name
    # ...transpose
    transformed = transformed.transpose(dims_orig)

    # add metadata
    transformed._metadata = geo_array._metadata
    _add_grid_mapping_metadata(transformed, to_grid_mapping)

    return transformed


def transform_vectors(u, v, to_grid_mapping, from_grid_mapping=None, \
        xt=None, yt=None, horizontal_coordinates=None):
    """ Transform vector field array into a new coordinate system and \
            interpolate values onto a new regular grid

    Assume the vector field is represented by an array of shape (2, Ny, Nx)

    Parameters
    ----------
    u, v : GeoArray or other DimArray instances
        x- and y- vector components
    to_grid_mapping : str or dict or cartopy.crs.CRS instance
        grid mapping onto which the transformation should be done
        str : PROJ.4 str or cartopy.crs.CRS class name
        dict : CF parameters
    from_grid_mapping : idem, optional
        original grid mapping, to be provided only when no grid_mapping 
        attribute is defined or when the axes are something else than 
        Longitude and Latitude
    xt, yt : array-like (1-D), optional
        new coordinates to interpolate the array on
        will be deduced as min and max of new coordinates if not provided
    horizontal_coordinates : sequence with 2 str, optional
        provide horizontal coordinates by name if not an instance of 
        Latitude, Longitude, X or Y

    Return
    ------
    transformed : GeoArray
        new 3-D GeoArray transformed and interpolated
    """ 
    if not isinstance(u, DimArray) or not isinstance(v, DimArray):
        raise TypeError("u and v must be DimArray instances")
    if not isinstance(u, GeoArray): 
        u = GeoArray(u) 
    if not isinstance(v, GeoArray): 
        v = GeoArray(v) 

    # consistency check between u and v
    assert u.axes == v.axes , "u and v must have the same axes"
    if from_grid_mapping is None and hasattr(u, 'grid_mapping'):
        assert hasattr(v, 'grid_mapping') and u.grid_mapping == v.grid_mapping, 'u and v must have the same grid mapping'

    # get grid mapping instances
    from_grid_mapping = _check_grid_mapping(from_grid_mapping, u)
    to_grid_mapping = _check_grid_mapping(to_grid_mapping)

    # find horizontal coordinates
    x0, y0 = _check_horizontal_coordinates(u, horizontal_coordinates)

    # Transform coordinates and prepare regular grid for interpolation
    xt, yt, x0_interp, y0_interp = transform_coords(from_grid_mapping, to_grid_mapping, x0, y0, xt, yt)

    # Transform vector components
    x0_2d, y0_2d = np.meshgrid(x0, y0)
    ut, vt = to_grid_mapping.transform_vectors(from_grid_mapping, x0_2d, y0_2d, u.values, v.values)

    if u.ndim == 2:
        newu = interp(u.values, x0.values, y0.values, x0_interp, y0_interp)
        ut = u._constructor(newu, [yt, xt])
        newv = interp(v.values, x0.values, y0.values, x0_interp, y0_interp)
        vt = v._constructor(newv, [yt, xt])

    else:
        # first reshape to 3-D components, flattening everything except vertical coordinates
        # TODO: optimize by computing and re-using weights?
        obj = stack([u, v], axis='vector_components', keys=['u','v'])
        obj = obj.group((x0.name, xt.name), reverse=True, insert=0) # 
        newvalues = []
        for k, suba in obj.iter(axis=0): # iterate over the first dimension
            newu = interp(suba.values[0], x0.values, y0.values, x0_interp, y0_interp)
            newv = interp(suba.values[0], x0.values, y0.values, x0_interp, y0_interp)
            newvalues.append(np.array([newu, newv]))

        # stack the arrays together
        newvalues = np.array(newvalues) # 4-D : grouped, vector_components, y, x
        grouped_obj = u._constructor(newvalues, [obj.axes[0], obj.axes[1], yt, xt])
        ut, vt = grouped_obj.ungroup(axis=0).swapaxes('vector_components',0)

    # add metadata
    ut._metadata = u._metadata
    vt._metadata = v._metadata
    _add_grid_mapping_metadata(ut, to_grid_mapping)
    _add_grid_mapping_metadata(vt, to_grid_mapping)

    return ut, vt
