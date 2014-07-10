""" Geo-Array transformation between various coordinate reference system
""" 
import warnings
import numpy as np
from dimarray.geo.geoarray import GeoArray, DimArray
from dimarray.geo.geoarray import X, Y, Z, Longitude, Latitude 
from dimarray import stack

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
    from dimarray.geo.crs import get_grid_mapping, CF_CRS

    if grid_mapping is None and geo_array is not None \
        and hasattr(geo_array, 'grid_mapping'):
            grid_mapping = geo_array.grid_mapping

    if grid_mapping is None: 
        raise ValueError("grid mapping not provided")

    # already a CRS instance : do nothing
    if isinstance(grid_mapping, ccrs.CRS):
        return grid_mapping

    # some more possibility offered, such as providing proj.4-like parameters
    else:
        grid_mapping = get_grid_mapping(grid_mapping)

    # now add a cf_params attribute if needed
    # NOTE: for now just do not force conversion from Cartopy
    # since the CF version is still experimental
    if not isinstance(grid_mapping, CF_CRS):
        try:
            crs = get_grid_mapping(grid_mapping, cf_conform=True)
            grid_mapping.cf_params = grid_mapping
        except:
            pass

    return grid_mapping


#
# Transformations involving changes in Coordinate Reference System
#
def _transform_coordinates(from_grid_mapping, to_grid_mapping, x0_2d, y0_2d, xt=None, yt=None):
    """ project coordinates

    Parameters
    ----------
    from_grid_mapping, to_grid_mapping 
    x0_2d, y0_2d : start coordinates, 2D
    xt, yt : ndarrays, optional
        transformed axes to interpolate on
        (otherwise they will be determined) 

    Return
    ------
    xt_2d, yt_2d : transformed 2-D coordinates
    xt, yt : regular 1-D coordinates to interpolate on
    """
    # Transform coordinates 
    #xt_2d, yt_2d = transform_coords(x0_2d, x0_2d, from_grid_mapping, to_grid_mapping)
    xt_2d, yt_2d = to_grid_mapping.transform_points(from_grid_mapping, x0_2d, x0_2d)

    # Interpolate onto a new regular grid while roughly conserving the steps
    if xt is None:
        xt = np.linspace(xt_2d.min(), xt_2d.max(), xt_2d.shape[1])
    if yt is None:
        yt = np.linspace(yt_2d.min(), yt_2d.max(), yt_2d.shape[0])

    assert xt.ndim == 1 and yt.ndim == 1, 'transformed coordinates must be 1-D'

    return xt_2d, yt_2d, xt, yt

#
#
#
def transform_scalars(geo_array, to_grid_mapping, from_grid_mapping=None, \
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
    from dimarray.compat.basemap import interp

    if not isinstance(geo_array, DimArray):
        raise TypeError("geo_array must be a DimArray instance")
    if not isinstance(geo_array, GeoArray):
        geo_array = GeoArray(geo_array) 

    # find horizontal coordinates
    x0, y0 = _check_horizontal_coordinates(geo_array, horizontal_coordinates)

    # transpose the array to shape .., y0, x0 (cartesian convention needed for meshgrid)
    dims_orig = geo_array.dims
    dims_new = [d for d in seld.dims if d not in [x0.name, y0.name]] + [y0.name, x0.name]
    geo_array = geo_array.transpose(dims_new) 

    #assert geo_array.dims.index(x0.name) > geo_array.axes[

    # get cartopy.crs.CRS instances
    from_grid_mapping = _check_grid_mapping(from_grid_mapping, geoarray)
    to_grid_mapping = _check_grid_mapping(from_grid_mapping)
    #from_grid_mapping = _get_grid_mapping(geoarray, from_grid_mapping)
    #to_grid_mapping = get_grid_mapping(to_grid_mapping)

    # Transform coordinates and prepare regular grid for interpolation
    x0_2d, y0_2d = np.meshgrid(x0.values, y0.values)
    xt_2d, yt_2d, xt_1d, yt_1d = _transform_coordinates(from_grid_mapping, to_grid_mapping, x0_2d, y0_2d, xt, yt)
    xt_2dr, yt_2dr = np.meshgrid(xt_1d, yt_1d) # regular 2-D grid

    # Define new axes
    # ...toward geodetic coordinates? (several standards seem to exist, cartopy uses "lonlat"
    if '+proj=latlong' in to_grid_mapping.proj4_init \
            or '+proj=lonlat' in to_grid_mapping.proj4_init:
        newaxx = Longitude(xt_1d)
        newaxy = Longitude(yt_1d)

    # ...make some projection
    else:
        newaxx = X(xt_1d)
        newaxy = Y(xt_1d)

    if geo_array.ndim == 2:
        newvalues = interp(geo_array.values, xt_2d, yt_2d, xt_2dr, yt_2dr)
        transformed = geo_array._constructor(newvalues, [newaxy, newaxx])

    else:
        # first reshape to 3-D, flattening everything except vertical coordinates
        # TODO: optimize by computing and re-using weights?
        obj = geo_array.group((x0.name, xt.name), reverse=True, insert=0)  
        newvalues = []
        for k, suba in obj.iter(axis=0): # iterate over the first dimension
            newval = interp(suba.values, xt_2d, yt_2d, xt_2dr, yt_2dr)
            newvalues.append(newval)

        # stack the arrays together
        newvalues = np.array(newvalues)
        grouped_obj = geo_array._constructor(newvalues, [obj.axes[0], newaxy, newaxx])
        transformed = grouped_obj.ungroup(axis=0)

    # reshape back
    # ...replace old axis names by new ones of the projection
    dims_orig = list(dims_orig)
    dims_orig[dims_orig.index(x0.name)] = newaxx.name
    dims_orig[dims_orig.index(y0.name)] = newaxy.name
    # ...transpose
    transformed = transformed.transpose(dims_orig)

    # add metadata
    transformed._metadata = geo_array._metadata
    try:
        cf_params = to_grid_mapping.cf_params
        transformed.grid_mapping = cf_params
    except:
        warnings.warn("cannot convert grid mapping to CF-conform metadata")
        if hasattr(transformed, 'grid_mapping'):
            del transformed.grid_mapping # remove old metadata

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
    to_grid_mapping = _check_grid_mapping(from_grid_mapping)

    # find horizontal coordinates
    x0, y0 = _check_horizontal_coordinates(geo_array, horizontal_coordinates)

    # Transform coordinates and prepare regular grid for interpolation
    x0_2d, y0_2d = np.meshgrid(x0.values, y0.values)
    xt_2d, yt_2d, xt_1d, yt_1d = _transform_coordinates(from_grid_mapping, to_grid_mapping, x0_2d, y0_2d, xt, yt)
    xt_2dr, yt_2dr = np.meshgrid(xt_1d, yt_1d) # regular 2-D grid

    # Transform vector components
    ut, vt = to_grid_mapping.transform_vectors(from_grid_mapping, x.values, y.values, u, v)

    if u.ndim == 3:
        newu = interp(u.values, xt_2d, yt_2d, xt_2dr, yt_2dr)
        ut = u._constructor(newu, [newaxy, newaxx])
        newv = interp(v.values, xt_2d, yt_2d, xt_2dr, yt_2dr)
        vt = v._constructor(newv, [newaxy, newaxx])

    else:
        # first reshape to 3-D components, flattening everything except vertical coordinates
        # TODO: optimize by computing and re-using weights?
        obj = stack([u, v], axis='vector_components', keys=['u','v'])
        obj = obj.group((x0.name, xt.name), reverse=True, insert=0) # 
        newvalues = []
        for k, suba in obj.iter(axis=0): # iterate over the first dimension
            newu = interp(suba.values[0], xt_2d, yt_2d, xt_2dr, yt_2dr)
            newv = interp(suba.values[0], xt_2d, yt_2d, xt_2dr, yt_2dr)
            newvalues.append(np.array([newu, newv]))

        # stack the arrays together
        newvalues = np.array(newvalues) # 4-D : grouped, vector_components, y, x
        grouped_obj = geo_array._constructor(newvalues, [obj.axes[0], obj.axes[1], newaxy, newaxx])
        ut, vt = grouped_obj.ungroup(axis=0).swapaxes('vector_components',0)

    # add metadata
    ut._metadata = u._metadata
    vt._metadata = v._metadata
    try:
        cf_params = to_grid_mapping.cf_params
        ut.grid_mapping = cf_params
        vt.grid_mapping = cf_params
    except:
        warnings.warn("cannot convert grid mapping to CF-conform metadata")
        if hasattr(ut, 'grid_mapping'): del ut.grid_mapping # remove old metadata
        if hasattr(vt, 'grid_mapping'): del vt.grid_mapping # remove old metadata

    return ut, vt
