""" Additional transformations
"""
import numpy as np
from dimarray.core import Axes, Axis

#from mpl_toolkits.basemap import interp
from dimarray.compat.basemap import interp

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

def interp1d_numpy(obj, values=None, axis=0, left=np.nan, right=np.nan):
    """ interpolate along one axis: wrapper around numpy's interp

    Parameters
    ----------
        obj : DimArray
        newaxis_values: 1d array, or Axis object
        newaxis_name, optional: `str` (axis name), required if newaxis is an array

    Returns
    -------
        interpolated data (n-d)
    """
    newaxis = Axis(values, axis)

    def interp1d(obj):
        """ 2-D interpolation function appled recursively on the object
        """
        xp = obj.axes[newaxis.name].values
        fp = obj.values
        result = np.interp(newaxis.values, xp, fp, left=left, right=right)
        return obj._constructor(result, [newaxis], **obj._metadata)

    result = apply_recursive(obj, (newaxis.name,), interp1d)
    return result.transpose(obj.dims) # transpose back to original dimensions

def interp1d(obj, values=None, axis=0, order=1, **kwargs):
    """ interpolate along one axis
    """
    if order == 1:
        return interp1d_numpy(obj, values, axis, **kwargs)

    else:
        raise NotImplementedError('order: '+repr(order))
        #return obj.reindex_axis(values, axis, method=method **kwargs)

def interp2d(obj, newaxes, dims=None, order=1):
    """ bilinear interpolation: wrapper around mpl_toolkits.basemap.interp

    Parameters
    ----------
        obj : DimArray
        newaxes: list of Axis object, or list of 1d arrays
        dims, optional: list of str (axis names), required if newaxes is a list of arrays

    Returns
    -------
        interpolated data (n-d)
    """

    # make sure input axes have the valid format
    newaxes = Axes._init(newaxes, dims) # valid format
    newaxes.sort(obj.dims) # re-order according to object's dimensions
    x, y = newaxes  # 2-d interpolation

    # make new grid 2-D
    x1, x1 = np.meshgrid(x.values, y.values, indexing='ij')

    def interp2d(obj, order=1):
        """ 2-D interpolation function appled recursively on the object
        """
        x0, y0 = obj.axes[x.name].values, obj.axes[y.name].values
        res = interp(obj.values, x0, y0, x1, y1, order=order)
        return obj._constructor(res, newaxes, **obj._metadata)

    result = apply_recursive(obj, (x.name, y.name), interp2d)
    return result.transpose(obj.dims) # transpose back to original dimensions
