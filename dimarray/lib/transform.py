""" Additional transformations
"""
import warnings
import numpy as np
from dimarray.core import Axes, Axis

#
# recursively apply a DimArray ==> DimArray transform
#
def apply_recursive(obj, dims, fun, *args, **kwargs):
    """ recursively apply a multi-axes transform

    dims   : dimensions to apply the function on
                Recursive call until dimensions match

    fun     : function to (recursively) apply, must returns a DimArray instance or something compatible
    *args, **kwargs: arguments passed to fun in addition to the axes
    """
    #
    # Check that dimensions are fine
    # 
    assert set(dims).issubset(set(obj.dims)), \
            "dimensions ({}) not found in the object ({})!".format(dims, obj.dims)

    #
    # If dims exactly matches: Done !
    # 
    if set(obj.dims) == set(dims):
        #assert set(obj.dims) == set(dims), "something went wrong"
        return fun(obj, *args, **kwargs)

    #
    # otherwise recursive call
    #

    # search for an axis which is not in dims
    i = 0
    while obj.axes[i].name in dims:
        i+=1 

    # make sure it worked
    assert obj.axes[i].name not in dims, "something went wrong"

    # now make a slice along this axis
    axis = obj.axes[i]

    # Loop over one axis
    data = []
    for axisval in axis.values: # first axis

        # take a slice of the data (exact matching)
        slice_ = obj.take(axisval, axis=axis.name)

        # apply the transform recursively
        res = apply_recursive(slice_, dims, fun, *args, **kwargs)

        data.append(res.values)

    newaxes = [axis] + res.axes # new axes
    data = np.array(data) # numpy array

    # automatically sort in the appropriate order
    new = obj._constructor(data, newaxes)
    return new

#
# INTERPOLATION
#

def interp1d_numpy(obj, values, axis=0, **kwargs):
    """ interpolate along one axis: wrapper around numpy's interp

    Parameters
    ----------
    obj : DimArray
    values : 1d array, or Axis object
    axis, optional : `str` (axis name), required if newaxis is an array

    Returns
    -------
    interpolated data (n-d)
    """
    warnings.warn(FutureWarning("Deprecated. Use DimArray.interp_axis"))
    return obj.interp_axis(values, axis=axis, **kwargs)

interp1d = interp1d_numpy


def interp2d(dim_array, newaxes, dims=(-2, -1), **kwargs):
    """ Two-dimensional interpolation

    Parameters
    ----------
    dim_array : DimArray instance
    newaxes : sequence of two array-like, or dict.
        axes on which to interpolate
    dims : sequence of two axis names or integer rank, optional
        Indicate dimensions which match `newaxes`.
        By default (-2, -1) (last two dimensions).
    **kwargs : passed to scipy.interpolate.RegularGridInterpolator
        method : 'nearest' or 'linear' (default)
        bounds_error : True by default
        fill_value : np.nan by default, but set to None to extrapolate outside bounds.

    Returns
    -------
    dim_array_int : DimArray instance
        interpolated array

    Examples
    --------

    >>> from dimarray import DimArray, interp2d
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 10])
    >>> a = DimArray([[0,0,1],[1,0.,0.]], [('y',y),('x',x)])
    >>> a
    dimarray: 6 non-null elements (0 null)
    0 / y (2): 0 to 10
    1 / x (3): 0 to 2
    array([[ 0.,  0.,  1.],
           [ 1.,  0.,  0.]])
    >>> newx = [0.5, 1.5]
    >>> newy = np.linspace(0,10,5)
    >>> ai = interp2d(a, [newy, newx])
    >>> ai
    dimarray: 10 non-null elements (0 null)
    0 / y (5): 0.0 to 10.0
    1 / x (2): 0.5 to 1.5
    array([[ 0.   ,  0.5  ],
           [ 0.125,  0.375],
           [ 0.25 ,  0.25 ],
           [ 0.375,  0.125],
           [ 0.5  ,  0.   ]])

    Use dims keyword argument if new axes order does not match array dimensions
    >>> (ai == interp2d(a, [newx, newy], dims=('x','y'))).all()
    True

    Out-of-bounds filled with NaN:
    >>> newx = [-1, 1]
    >>> newy = [-5, 0, 10]
    >>> interp2d(a, [newy, newx], bounds_error=False)
    dimarray: 2 non-null elements (4 null)
    0 / y (3): -5 to 10
    1 / x (2): -1 to 1
    array([[ nan,  nan],
           [ nan,   0.],
           [ nan,   0.]])

    Nearest neighbor interpolation and out-of-bounds extrapolation
    >>> interp2d(a, [newy, newx], method='nearest', bounds_error=False, fill_value=None)
    dimarray: 6 non-null elements (0 null)
    0 / y (3): -5 to 10
    1 / x (2): -1 to 1
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 1.,  0.]])
    """
    from scipy.interpolate import RegularGridInterpolator

    # back compatibility
    _order = kwargs.pop('order', None)
    if _order is not None:
        warnings.warn('order is deprecated, use method instead', DeprecationWarning)
        if _order == 3: 
            warnings.warn('cubic not supported. Switch to linear.', DeprecationWarning)
        kwargs['method'] = {0:'nearest', 1:'linear', 3:'linear'}[_order]

    _clip = kwargs.pop('clip', None)
    if _clip is not None:
        warnings.warn('clip is deprecated, set bounds_error=False and fill_value=None \
                      for similar results (see scipy.interpolate.RegularGridInterpolator)', DeprecationWarning)
        if _clip is True:
            kwargs['bounds_error'] = False
            kwargs['fill_value'] = None

    # provided as a dictionary
    if isinstance(newaxes, dict):
        dims = newaxes.keys()
        newaxes = newaxes.values()

    if not isinstance(newaxes, list) or isinstance(newaxes, tuple):
        raise TypeError("newaxes must be a sequence of axes to interpolate on")

    if len(newaxes) != 2:
        raise ValueError("must provide two axis values to interpolate on")

    if len(dims) != 2: 
        raise ValueError("must provide two axis names to interpolate on")

    x0 = dim_array.axes[dims[0]]
    y0 = dim_array.axes[dims[1]]

    # new axes
    xi, yi = newaxes
    xi = Axis(xi, x0.name) # convert to Axis
    yi = Axis(yi, y0.name)


    ## transpose the array to shape .., x0, y0
    dims_orig = dim_array.dims
    dims_new = [d for d in dim_array.dims if d not in [x0.name, y0.name]] + [x0.name, y0.name]
    dim_array = dim_array.transpose(dims_new) 

    _xi2, _yi2 = np.meshgrid(xi.values, yi.values, indexing='ij') # requires 2-D grid
    _new_points = np.array([_xi2.flatten(), _yi2.flatten()]).T
    def _interp_map(x, y, z):
        f = RegularGridInterpolator((x, y), z, **kwargs)
        return f(_new_points).reshape(_xi2.shape)

    if dim_array.ndim == 2:
        newvalues = _interp_map(x0.values, y0.values, dim_array.values)
        dim_array_int = dim_array._constructor(newvalues, [xi, yi])

    else:
        # first reshape to 3-D, flattening everything except horizontal_coordinates coordinates
        dim_array = dim_array.flatten((x0.name, y0.name), reverse=True, insert=0)  
        newvalues = []
        for k, suba in dim_array.iter(axis=0): # iterate over the first dimension
            newval = _interp_map(x0.values, y0.values, suba.values)
            newvalues.append(newval)

        # stack the arrays together
        newvalues = np.array(newvalues)
        flattened_dim_array = dim_array._constructor(newvalues, [dim_array.axes[0], xi, yi])
        dim_array_int = flattened_dim_array.unflatten(axis=0)

    # reshape back
    # ...replace old axis names by new ones of the projection
    dims_orig = list(dims_orig)
    # ...transpose
    dim_array_int = dim_array_int.transpose(dims_orig)

    # add metadata
    dim_array_int.attrs.update(dim_array.attrs)

    return dim_array_int
