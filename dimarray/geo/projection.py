""" Geo-Array transformation between various coordinate reference system
""" 
import warnings
import numpy as np
from dimarray.geo.geoarray import GeoArray, DimArray, Coordinate, Axis
from dimarray.geo.geoarray import X, Y, Z, Longitude, Latitude 
from dimarray import stack
from scipy.interpolate import RegularGridInterpolator

#
# share private functions
#
def _check_horizontal_coordinates(geo_array, horizontal_coordinates=None, add_grid_mapping=True):
    """ return horizontal coordinates
    """
    if horizontal_coordinates is not None: 
        assert len(horizontal_coordinates) == 2, "horizontal_coordinates must be a sequence of length 2"
        x0nm, y0nm = horizontal_coordinates
        x0 = geo_array.axes[x0nm]
        y0 = geo_array.axes[y0nm]

    else:
        xs = filter(lambda x: isinstance(x, X), geo_array.axes)
        ys = filter(lambda x: isinstance(x, Y), geo_array.axes)

        if len(xs) > 0:
            x0 = xs[0]
            assert len(xs) == 1, "Several X-coordinates found." # this should not happen
        else:
            warnings.warn("Could not find X-coordinate among GeoArray axes. Use dimension 1 by default.")
            x0 = geo_array.axes[1]

        if len(ys) > 0:
            y0 = ys[0]
            assert len(ys) == 1, "Several Y-coordinates found." # this should not happen
        else:
            warnings.warn("Could not find Y-coordinate among GeoArray axes. Use dimension 0 by default.")
            y0 = geo_array.axes[0]


    # Add grid mapping to GeoArray if Coordinates are Geodetic (long/lat)
    if add_grid_mapping and geo_array.grid_mapping is None \
        and isinstance(x0, Longitude) and isinstance(y0, Latitude):
            geo_array.grid_mapping = {'grid_mapping_name':'latitude_longitude'}

    return x0, y0


def _get_crs(grid_mapping=None, geo_array=None):
    """ Get CRS instance corresponding to a grid mapping

    Parameters
    ----------
    grid_mapping : PROJ.4 str or CF dict or CRS instance
    geo_array : geoarray
    add a "cf_params" attribute if possible, to help add metadata
    """
    import cartopy.crs as ccrs
    from dimarray.geo.crs import get_crs

    if isinstance(grid_mapping, ccrs.CRS):
        crs_obj = grid_mapping

    else:
        if grid_mapping is None and geo_array is not None \
            and hasattr(geo_array, 'grid_mapping'):
                grid_mapping = geo_array.grid_mapping

        if grid_mapping is None: 
            raise ValueError("grid mapping not provided")

        crs_obj = get_crs(grid_mapping)

    return crs_obj

def _create_Axes_with_metadata(xt, yt, grid_mapping):
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
            xt.attrs = meta

        else:
            xt = X(xt, 'x')

        # Make actual Coordinate axes
        if hasattr(grid_mapping, '_y_metadata'):
            meta = grid_mapping._y_metadata
            name = meta.pop('name', 'y')
            yt = Y(yt, name)
            yt.attrs = meta

        else:
            yt = Y(yt, 'y')

    return xt, yt

#
# Transformations involving changes in Coordinate Reference System
#
def _inverse_transform_coords(from_crs, to_crs, xt=None, yt=None, x0=None, y0=None):
    """ Provide 2-D grid in from_crs whose transform in to_crs is regular

    This function is called by transform and transform_vectors, to ease interpolation
    of the arrays onto a regular grid in to_crs.

    Parameters
    ----------
    from_crs, to_crs : cartopy.crs.CRS instances
        coordinate systems ("from" and "to" refer to calling function)
    xt, yt : 1-D, array-like, optional
        target regular coordinates in to_crs to inverse-transform into from_crs
        If not provided, xt and yt will be determined by first gridding and 
        and transforming x0 and y0 from from_crs into to_crs and 
        looking for min / max bounds. 
    x0, y0 : 1-D array-like, optional
        original, regular coordinates in from_crs
        only used if xt, yt are not provided

    Returns
    -------
    x0_interp, y0_interp : 2-D ndarrays 
        coordinates in from_crs whose transform in to_crs yields meshgrid(xt, yt)
    xt, yt : Axis instances
        1-D coordinates in to_crs, Axis version of input parameter xt and yt

    Examples
    --------
    >>> from dimarray.compat import cartopy_crs as ccrs
    >>> from_crs = ccrs.NorthPolarStereo(true_scale_latitude=71.)
    >>> to_crs = ccrs.Geodetic()
    >>> xt = np.linspace(-50, 10, 5) # desired lon range
    >>> yt = np.linspace(60, 85, 5)  # desired lat range
    >>> x0_i, y0_i, Xt, Yt  = _inverse_transform_coords(from_crs, to_crs, xt, yt)
    >>> x0_i.shape == x0_i.shape == (5,5)
    True
    >>> from numpy.testing import assert_allclose
    >>> trans = to_crs.transform_points(from_crs, x0_i, y0_i)
    >>> assert_allclose((trans[:,:,0], trans[:,:,1]), np.meshgrid(xt, yt))
    """
    # Determine xt and yt
    if xt is None or yt is None:

        assert x0 is not None and y0 is not None, "x0 and y0 must be provided"
        x0 = np.asarray(x0)
        y0 = np.asarray(y0)
        assert x0.ndim ==  1 and y0.ndim == 1
        x0_2d, y0_2d = np.meshgrid(x0, y0) # make 2-D coordinates
        pts_xyz = to_crs.transform_points(from_crs, x0_2d, y0_2d)
        xt_2d, yt_2d = pts_xyz[..., 0], pts_xyz[..., 1]

        # Make regular grid in new coordinate system for interpolation
        if xt is None:
            xt = np.linspace(xt_2d.min(), xt_2d.max(), x0.size) # keep about the same size
        if yt is None:
            yt = np.linspace(yt_2d.min(), yt_2d.max(), y0.size) # keep about the same size

    assert (isinstance(xt, Axis) or xt.ndim == 1) \
            and (isinstance(yt, Axis) or yt.ndim == 1), 'transformed coordinates must be 1-D'

    # Transform back to from_crs
    # ...make it 2D
    xt_2dr, yt_2dr = np.meshgrid(xt, yt) # regular 2-D grid
    # ...transform
    pts_xyz = from_crs.transform_points(to_crs, xt_2dr, yt_2dr)
    x0_int, y0_int = pts_xyz[..., 0], pts_xyz[..., 1]

    # Create new axes with appropriate metadata (unless alread Axis instances are provided)
    if not (isinstance(xt, Axis) and isinstance(yt, Axis)):
        xt, yt = _create_Axes_with_metadata(xt, yt, to_crs)

    return x0_int, y0_int, xt, yt

#
#
#
def transform(geo_array, to_crs, from_crs=None, \
        xt=None, yt=None, **kwargs):
    """ Transform scalar field array into a new coordinate system and \
            interpolate values onto a new regular grid

    Parameters
    ----------
    geo_array : GeoArray or other DimArray instance
    to_crs : str or dict or cartopy.crs.CRS instance
        grid mapping onto which the transformation should be done
        str : PROJ.4 str or cartopy.crs.CRS class name
        dict : CF parameters
    from_crs : idem, optional
        original grid mapping. Can be omitted if the grid_mapping attribute
        already contains the appropriate information, or if the horizontal
        coordinates are longitude and latitude.
    xt, yt : array-like (1-D), optional
        new coordinates to interpolate the array on
        will be deduced as min and max of new coordinates if not provided
    **kwargs: passed to scipy.interpolate.RegularGridInterpolator
        error_bounds (True), method (linear), fill_value (np.nan)

    Returns
    -------
    transformed : GeoArray
        new GeoArray transformed
        Attempt is made to document the projection with CF-conform metadata

    Examples
    --------
    """ 
    # local import since it's quite heavy
    if not isinstance(geo_array, DimArray):
        raise TypeError("geo_array must be a DimArray instance")
    if not isinstance(geo_array, GeoArray):
        geo_array = GeoArray(geo_array) 

    # back compat
    _masked = kwargs.pop('masked', None)
    if _masked is not None: 
        warnings.warn('masked is deprecated.', DeprecationWarning)

    # find horizontal coordinates
    x0, y0 = _check_horizontal_coordinates(geo_array)

    # transpose the array to shape .., y0, x0 (cartesian convention needed for meshgrid)
    dims_orig = geo_array.dims
    dims_new = [d for d in geo_array.dims if d not in [x0.name, y0.name]] + [y0.name, x0.name]
    geo_array = geo_array.transpose(dims_new) 

    #assert geo_array.dims.index(x0.name) > geo_array.axes[

    # get cartopy.crs.CRS instances
    from_crs = _get_crs(from_crs, geo_array)
    to_crs = _get_crs(to_crs)

    # Transform coordinates and prepare regular grid for interpolation
    x0_interp, y0_interp, xt, yt = _inverse_transform_coords(from_crs, to_crs, xt, yt, x0, y0)

    _new_points = np.array([y0_interp.flatten(), x0_interp.flatten()]).T
    def _interp_map(x, y, z):
        f = RegularGridInterpolator((y, x), z, **kwargs)
        return f(_new_points).reshape(x0_interp.shape)



    if geo_array.ndim == 2:
        #newvalues = interp(geo_array.values, xt_2d, yt_2d, xt_2dr, yt_2dr)
        #newvalues = interp(geo_array.values, x0.values, y0.values, x0_interp, y0_interp, masked=masked)
        newvalues = _interp_map(x0.values, y0.values, geo_array.values)
        transformed = geo_array._constructor(newvalues, [yt, xt])

    else:
        # first reshape to 3-D, flattening everything except horizontal_coordinates coordinates
        # TODO: optimize by computing and re-using weights?
        obj = geo_array.flatten((x0.name, y0.name), reverse=True, insert=0)  
        newvalues = []
        for k, suba in obj.iter(axis=0): # iterate over the first dimension
            #newval = interp(suba.values, xt_2d, yt_2d, xt_2dr, yt_2dr)
            #newval = interp(suba.values, x0.values, y0.values, x0_interp, y0_interp, masked=masked)
            newval = _interp_map(x0.values, y0.values, suba.values)
            newvalues.append(newval)

        # stack the arrays together
        newvalues = np.array(newvalues)
        flattened_obj = geo_array._constructor(newvalues, [obj.axes[0], yt, xt])
        transformed = flattened_obj.unflatten(axis=0)

    # reshape back
    # ...replace old axis names by new ones of the projection
    dims_orig = list(dims_orig)
    dims_orig[dims_orig.index(x0.name)] = xt.name
    dims_orig[dims_orig.index(y0.name)] = yt.name
    # ...transpose
    transformed = transformed.transpose(dims_orig)

    # add metadata
    transformed.attrs.update(geo_array.attrs)

    _add_grid_mapping_metadata(transformed, to_crs)

    return transformed


def transform_vectors(u, v, to_crs, from_crs=None, \
        xt=None, yt=None, **kwargs):
    """ Transform vector field array into a new coordinate system and \
            interpolate values onto a new regular grid

    Assume the vector field is represented by an array of shape (2, Ny, Nx)

    Parameters
    ----------
    u, v : GeoArray or other DimArray instances
        x- and y- vector components
    to_crs : str or dict or cartopy.crs.CRS instance
        grid mapping onto which the transformation should be done
        str : PROJ.4 str or cartopy.crs.CRS class name
        dict : CF parameters
    from_crs : idem, optional
        original grid mapping. Can be omitted if the grid_mapping attribute
        already contains the appropriate information, or if the horizontal
        coordinates are longitude and latitude.
    xt, yt : array-like (1-D), optional
        new coordinates to interpolate the array on
        will be deduced as min and max of new coordinates if not provided
    **kwargs: passed to scipy.interpolate.RegularGridInterpolator
        error_bounds (True), method (linear), fill_value (np.nan)

    Returns
    -------
    transformed : GeoArray
        new 3-D GeoArray transformed and interpolated
    """ 
    # back compat
    _masked = kwargs.pop('masked', None)
    if _masked is not None: 
        warnings.warn('masked is deprecated.', DeprecationWarning)

    if not isinstance(u, DimArray) or not isinstance(v, DimArray):
        raise TypeError("u and v must be DimArray instances")
    if not isinstance(u, GeoArray): 
        u = GeoArray(u) 
    if not isinstance(v, GeoArray): 
        v = GeoArray(v) 

    # consistency check between u and v
    assert u.axes == v.axes , "u and v must have the same axes"
    if from_crs is None and hasattr(u, 'grid_mapping'):
        assert hasattr(v, 'grid_mapping') and u.grid_mapping == v.grid_mapping, 'u and v must have the same grid mapping'

    # get grid mapping instances
    from_crs = _get_crs(from_crs, u)
    to_crs = _get_crs(to_crs)

    # find horizontal coordinates
    x0, y0 = _check_horizontal_coordinates(u)

    # Transform coordinates and prepare regular grid for interpolation
    x0_interp, y0_interp, xt, yt = _inverse_transform_coords(from_crs, to_crs, xt, yt, x0, y0)

    # Transform vector components
    x0_2d, y0_2d = np.meshgrid(x0, y0)

    _new_points = np.array([y0_interp.flatten(), x0_interp.flatten()]).T
    def _interp_map(x, y, z):
        f = RegularGridInterpolator((y, x), z, **kwargs)
        return f(_new_points).reshape(x0_interp.shape)

    _constructor = u._constructor 
    if u.ndim == 2:
        # First transform vector components onto the new coordinate system
        _ut, _vt = to_crs.transform_vectors(from_crs, x0_2d, y0_2d, u.values, v.values) 
        # Then interpolate onto regular grid
        #_ui = interp(_ut, x0.values, y0.values, x0_interp, y0_interp, masked=masked)
        _ui = _interp_map(x0.values, y0.values, _ut)
        ut = _constructor(_ui, [yt, xt])
        #_vi = interp(_vt, x0.values, y0.values, x0_interp, y0_interp, masked=masked)
        _vi = _interp_map(x0.values, y0.values, _vt)
        vt = _constructor(_vi, [yt, xt])

    else:
        # first reshape to 3-D components, flattening everything except horizontal coordinates
        # TODO: optimize by computing and re-using weights?
        obj = stack([u, v], axis='vector_components', keys=['u','v'])
        obj = obj.flatten(('vector_components', x0.name, y0.name), reverse=True, insert=0) # 
        newvalues = []
        for k, suba in obj.iter(axis=0): # iterate over the first dimension
            # First transform vector components onto the new coordinate system
            _ut, _vt = to_crs.transform_vectors(from_crs, x0_2d, y0_2d, suba.values[0], suba.values[1]) 
            # Then interpolate onto regular grid
            #_ui = interp(_ut, x0.values, y0.values, x0_interp, y0_interp, masked=masked)
            _ui = _interp_map(x0.values, y0.values, _ut)
            #_vi = interp(_vt, x0.values, y0.values, x0_interp, y0_interp, masked=masked)
            _vi = _interp_map(x0.values, y0.values, _vt)
            newvalues.append(np.array([_ui, _vi]))

        # stack the arrays together
        newvalues = np.array(newvalues) # 4-D : flattened, vector_components, y, x
        flattened_obj = _constructor(newvalues, [obj.axes[0], obj.axes[1], yt, xt])
        ut, vt = flattened_obj.unflatten(axis=0).swapaxes('vector_components',0)

    # add metadata
    ut.attrs.update(u.attrs)
    vt.attrs.update(v.attrs)

    _add_grid_mapping_metadata(ut, to_crs)
    _add_grid_mapping_metadata(vt, to_crs)

    return ut, vt

def _add_grid_mapping_metadata(a, crs):
    """ add grid_mapping metadata to an array
    """
    # for now, just delete grid_mapping attribute, if any
    # TODO: systematic conversion from crs to CF-parameters
    #if hasattr(a, 'grid_mapping'): del a.grid_mapping # remove old metadata
    if 'grid_mapping' in a.__dict__: del a.grid_mapping # remove old metadata
