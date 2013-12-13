import numpy as np
from functools import partial
from axes import Axis, Axes

#
# Technicalities to apply numpy methods
#
class NumpyDesc(object):
    """ to apply a numpy function which reduces an axis
    """
    def __init__(self, apply_method, numpy_method, **kwargs):
	"""
	apply_method: as the string name of the bound method to consider
        numpy_method: as a string name of the numpy function to apply
	**kwargs    : default values for keyword arguments to numpy_method
	"""
	assert type(apply_method) is str, "can only provide method name as a string"
	assert type(numpy_method) is str, "can only provide method name as a string"
	self.numpy_method = numpy_method
	self.apply_method = apply_method
	self.kwargs = kwargs

    def __get__(self, obj, cls=None):
	"""
	"""
	# convert self.apply to an actual function
	apply_method = getattr(obj, self.apply_method) 
	newmethod = partial(apply_method, self.numpy_method, **self.kwargs)
	#newmethod = deco_numpy(apply_method, self.numpy_method, **self.kwargs)

	# Now replace the doc string with "apply" docstring
	newmethod.__doc__ = apply_method.__doc__.replace("`func`","`"+self.numpy_method+"`")

	return newmethod

#
# Define weighted mean/std/var
#

def weighted_mean(obj, axis=0, skipna=True, weights='axis', out=None):
    """ mean over an axis or group of axes, possibly weighted 

    parameters:
    ----------
	- axis    : int, str, tuple: axis or group of axes to apply the transform on
	- skipna  : remove nans prior to transformation?
	- weights : if weights, perform a weighted mean (see get_weights method)
		    the default behaviour ("axis") is too look in individual axes 
		    whether they have a not-None weight attribute
    
    returns:
    --------
	- Dimarray or scalar, consistently with ndarray behaviour
    """
    # Proceed to a weighted mean
    if weights:
	weights = obj.get_weights(weights, axis=axis, fill_nans=skipna)

    # if no weights, just apply numpy's mean
    if not weights:
	return obj.apply("mean", axis=axis, skipna=skipna, out=out)

    # weighted mean
    sum_values = (obj*weights).apply("sum", axis=axis, skipna=skipna, out=out)
    sum_weights = weights.apply("sum", axis=axis, skipna=skipna, out=out)
    return sum_values / sum_weights

def weighted_var(obj, axis=0, skipna=True, weights="axis", ddof=0, out=None):
    """ standard deviation over an axis or group of axes, possibly weighted 

    parameters:
    ----------
	- axis    : int, str, tuple: axis or group of axes to apply the transform on
	- skipna  : remove nans prior to transformation?
	- weights : if weights, perform a weighted var (see get_weights method)
		    the default behaviour ("axis") is too look in individual axes 
		    whether they have a not-None weight attribute
	- ddof    : "Delta Degrees of Freedom": the divisor used in the calculation is
		    ``N - ddof``, where ``N`` represents the number of elements. By default `ddof` is zero.
		    Note ddof is ignored when weights are used
    
    returns:
    --------
	- Dimarray or scalar, consistently with ndarray behaviour
    """
    # Proceed to a weighted var
    if weights:
	weights = obj.get_weights(weights, axis=axis, fill_nans=skipna)

    # if no weights, just apply numpy's var
    if not weights:
	return obj.apply("var", axis=axis, skipna=skipna, ddof=ddof, out=out)

    # weighted mean
    mean = obj.mean(axis=axis, skipna=skipna, weights=weights, out=out)
    dev = (obj-mean)**2
    return dev.mean(axis=axis, skipna=skipna, weights=weights, out=mean)

def weighted_std(obj, *args, **kwargs):
    """ alias for a.var()**0.5: see `var` method for doc
    """
    return obj.var(*args, **kwargs)**0.5

#
# recursively apply a Dimarray ==> Dimarray transform
#
def apply_recursive(obj, dims, fun, *args, **kwargs):
    """ recursively apply a multi-axes transform

    dims   : dimensions to apply the function on
		Recursive call until dimensions match

    fun     : function to (recursively) apply, must returns a Dimarray instance or something compatible
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
	slice_ = obj.xs_axis(axisval, axis=axis.name, method="index")
	#slice_ = obj.xs(**{ax.name:axisval, "method":"index"})

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

def interp1d_numpy(obj, values=None, axis=0, left=None, right=None):
    """ interpolate along one axis: wrapper around numpy's interp

    input:
	newaxis_values: 1d array, or Axis object
	newaxis_name, optional: `str` (axis name), required if newaxis is an array

    output:
	interpolated data (n-d)
    """
    newaxis = Axis(values, axis)

    def interp1d(obj, order=1):
	""" 2-D interpolation function appled recursively on the object
	"""
	xp = obj.axes[newaxis.name].values
	fp = obj.values
	result = np.interp(newaxis.values, xp, fp, left=left, right=right)
	return obj._constructor(result, [newaxis], **obj._metadata)

    result = apply_recursive(obj, (newaxis.name,), interp1d)
    return result.transpose(obj.dims) # transpose back to original dimensions


def interp2d_mpl(obj, newaxes, axes=None, order=1):
    """ bilinear interpolation: wrapper around mpl_toolkits.basemap.interp

    input:
	newaxes: list of Axis object, or list of 1d arrays
	axes, optional: list of str (axis names), required if newaxes is a list of arrays

    output:
	interpolated data (n-d)
    """
    from mpl_toolkits.basemap import interp

    # make sure input axes have the valid format
    newaxes = Axes.from_list(newaxes, axes) # valid format
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
