import numpy as np
from functools import partial
from axes import Axis, Axes, GroupedAxis

def apply(obj, func, axis=None, skipna=True, args=(), **kwargs):
    """ Apply `func` to Dimarray
    
    Generic method to apply along-axis or single-argument ufunc numpy method

    parameters:
    ----------
    - func: numpy function (can be provided by name)

    - axis    : None, int, str, tuple
		axis or group of axes to apply the transform on.
		- None : applies the transform on the flattened array, 
		         consistently with numpy.
		- `int`: axis position 
		- `str`: axis name
		- `tuple`: group of axes to flatten before applying the 
		           transform on
		Default is None

    - skipna  : If True, call corresponding np.ma method using NaN as mask 
		to convert into a MaskedArray.
		For "sum","max","min" the corresponding 
		`nan` numpy function is called.

    - args    : variable list of arguments before "axis"
    - kwargs  : variable dict of keyword arguments after "axis"
    
    returns:
    --------
    - Dimarray, or scalar (if `axis` is None)

    NOTE: if an axis has weights, `mean`, `std` and `var` will use these weights
    """
    #
    # For ufunc, the axes remain the same
    #
    if _is_ufunc(func):
	if type(func) is str:
	    func = getattr(np, func)
	return obj._constructor(func(obj.values), obj.axes.copy())

    #
    # Functions that operate along-axis
    #

    # Deal with `axis` parameter, whether `int`, `str` or `tuple`
    obj, idx, name = _deal_with_axis(obj, axis)

    # Apply numpy function, dealing with NaNs
    result = _apply_nans(obj.values, func, axis=idx, skipna=skipna, args=(), **kwargs)

    if type(func) is str:
	funcname = func
	func = getattr(np, func)
    else:
	funcname = func.__name__

    # If `axis` was None (operations on the flattened array), just returns the numpy array
    if axis is None or not isinstance(result, np.ndarray):
	return result

    #
    # New axes
    #
    # standard case: collapsed axis
    if np.ndim(result) == obj.ndim - 1:
	newaxes = [ax for ax in obj.axes if ax.name != name]

    # cumulative functions: axes remain unchanged
    elif funcname in ('cumsum','cumprod','gradient'):
	newaxes = obj.axes.copy() 

    # diff: reduce axis size by one
    elif funcname == "diff":
	oldaxis = obj.axes[idx]
	newaxis = oldaxis[1:]  # assume backward differencing 
	newaxes = obj.axes.copy() 
	newaxes[idx] = newaxis

    # axes do not fit for some reason
    else:
	raise Exception("cannot find new axes for this transformation: "+repr(funcname))

    newobj = obj._constructor(result, newaxes, **obj._metadata)

    # add stamp
    stamp = "{transform}({axis})".format(transform=funcname, axis=str(obj.axes[idx]))
    newobj._metadata_stamp(stamp)

    return newobj

def _apply_nans(values, funcname, axis=None, skipna=True, args=(), **kwargs):
    """ apply a numpy transform on numpy array but dealing with NaNs
    """
    # Deal with NaNs
    if skipna:

	# replace with the optimized numpy function if existing
	if funcname in ("sum","max","min","argmin","argmax"):
	    funcname = "nan"+funcname
	    module = np

	# otherwise convert to MaskedArray if needed
	else:
	    values = to_MaskedArray(values)
	    module = np.ma
    
    # use basic numpy functions 
    else:
	module = np 

    # get actual function if provided as str (the default)
    if type(funcname) is str:
	func = getattr(module, funcname)

    # but also accept functions provided as functions
    else:
	func = funcname
	funcname = func.__name__

    # Call functions
    kwargs['axis'] = axis
    result = func(values, *args, **kwargs) 

    # otherwise, fill NaNs back in
    if np.ma.isMaskedArray(result):
	result = result.filled(np.nan)

    return result

def to_MaskedArray(values):
    """ convert a numpy array to MaskedArray
    """
    # fast-check for NaNs, thanks to
    # http://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
    if np.isnan(np.min(values)):
	mask = np.isnan(values)
    else:
	mask = False
    values = np.ma.array(values, mask=mask)
    return values

def _is_ufunc(func):
    """ check whether a function is a ufun 
    (e.g. `sin`, `sqrt` etc...: functions that operate element by element on whole arrays.)
    """
    if type(func) is str:
	func = getattr(np, func)
    return isinstance(func, np.ufunc)

def _deal_with_axis(obj, axis):
    """ deal with the `axis` parameter 

    input:
	obj: Dimarray object
	axis: `int` or `str` or `tuple`

    return:
	newobj: reshaped obj if axis is tuple otherwise obj
	idx   : axis index
	name  : axis name
    """
    # before applying the function on the collapsed array
    if type(axis) is tuple:
	newobj = obj.group(axis)
	idx = obj.dims.index(axis[0])  # position where the new axis has been inserted
	ax = newobj.axes[idx] 
	name = ax.name

	# checking
	assert isinstance(ax, GroupedAxis) and ax.axes[0].name == axis[0], "problem when grouping axes"

    else:
	newobj = obj
	idx, name = obj._get_axis_info(axis) 

    return newobj, idx, name

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
	newmethod.__doc__ = newmethod.__doc__.replace("- func: numpy function (can be provided by name)\n\n","")

	return newmethod

cumsum = NumpyDesc("apply", "cumsum", axis=-1)
cumsum.__doc__ = cumsum.__doc__.replace("Default is None","Default is -1 (last axis)")

cumprod = NumpyDesc("apply", "cumprod", axis=-1)
cumprod.__doc__ = cumprod.__doc__.replace("Default is None","Default is -1 (last axis)")

def diff(obj, n=1, axis=-1, scheme="backward", keepaxis=False):
    """ Analogous to numpy's diff

    Calculate the n-th order discrete difference along given axis.

    The first order difference is given by ``out[n] = a[n+1] - a[n]`` along
    the given axis, higher order differences are calculated by using `diff`
    recursively.

    Parameters
    ----------
    n : int, optional
	The number of times values are differenced.

    axis : int, str, tuple (optional)
	The axis along which the difference is taken, default is the last
	axis, for consistency with numpy

    scheme: str, determines the values of the resulting axis
	    "forward" : diff[i] = x[i+1] - x[i]
	    "backward": diff[i] = x[i] - x[i-1]
	    "centered": diff[i] = x[i+1/2] - x[i-1/2]
	    default is "backward"

    keepaxis: bool, if True, keep the initial axis by padding with NaNs
	      Only compatible with "forward" or "backward" differences

    rate    : bool, default `False`: use axis spacing: TO DO

    Returns
    -------
    diff : Dimarray
	The `n` order differences. The shape of the output is the same as `a`
	except along `axis` where the dimension is smaller by `n`.
    """
    # If `axis` is None (operations on the flattened array), just returns the numpy array
    if axis is None:
	return np.diff(obj.values, n=n, axis=None)

    # Deal with `axis` parameter, whether `int`, `str` or `tuple`
    # possibly grouping dimensions if axis is tuple
    obj, idx, name = _deal_with_axis(obj, axis)

    # Recursive call if n > 1
    if n > 1:
	obj = obj.diff(n=n-1, axis=idx, scheme=scheme, keepaxis=keepaxis)
	n = 1

    # n = 1
    assert n == 1, "n must be integer greater or equal to one"

    # Compute differences
    result = np.diff(obj.values, axis=idx)

    # Old axis along diff
    oldaxis = obj.axes[idx]

    # forward differencing
    if scheme == "forward":

	# keep axis: pad last element with NaNs
	if keepaxis:
	    result = _append_nans(result, axis=idx)
	    newaxis = oldaxis.copy()

	# otherwise just shorten the axis
	else:
	    newaxis = oldaxis[:-1]

    elif scheme == "backward":

	# keep axis: pad first element with NaNs
	if keepaxis:
	    result = _append_nans(result, axis=idx, first=True)
	    newaxis = oldaxis.copy()

	# otherwise just shorten the axis
	else:
	    newaxis = oldaxis[1:]

    elif scheme == "centered":

	# keep axis: central difference + forward/backward diff at the edges
	if keepaxis:
	    #indices = range(oldaxis.size)
	    raise ValueError("keepaxis=True is not compatible with centered differences")
	    #central = obj.values.take(indices[2:], axis=idx) \
	    #        -  obj.values.take(indices[:-2], axis=idx)
	    #start = obj.values.take([1], axis=idx) \
	    #        -  obj.values.take([0], axis=idx)
	    #end = obj.values.take([-1], axis=idx) \
	    #        -  obj.values.take([-2], axis=idx)
	    #result = np.concatenate((start, central, end), axis=idx)
	    #newaxis = oldaxis.copy()

	else:
	    axisvalues = 0.5*(oldaxis.values[:-1]+oldaxis.values[1:])
	    newaxis = Axis(axisvalues, name)

    else:
	raise ValueError("scheme must be one of 'forward', 'backward', 'central', got {}".format(scheme))

    newaxes = obj.axes.copy() 
    newaxes[idx] = newaxis
    newobj = obj._constructor(result, newaxes, **obj._metadata)

    return newobj

def _append_nans(result, axis, first=False):
    """ insert a slice of NaNs at the front of an array along axis
    or append the slice if append is True
    """
    nan_slice = np.empty_like(result.take([0], axis=axis)) # make a slice ...
    nan_slice.fill(np.nan) # ...filled with NaNs

    # Insert as first element
    if first:
	result = np.concatenate((nan_slice, result), axis=axis)

    # Append
    else:
	result = np.concatenate((result, nan_slice), axis=axis)

    return result

#
# Special behaviour for argmin and argmax: return axis values instead of integer position
#
def argmin(obj, axis=None, skipna=True):
    """ similar to numpy's argmin, but return axis values instead of integer position
    """
    obj, idx, name = _deal_with_axis(obj, axis)
    res = obj.apply('argmin', axis=idx, skipna=skipna)

    # along axis: single axis value
    if axis is not None: # res is Dimarray
	res.values = obj.axes[idx].values[res.values] 
	return res

    # flattened array: tuple of axis values
    else: # res is ndarray
	res = np.unravel_index(res, obj.shape)
	return tuple(obj.axes[i].values[v] for i, v in enumerate(res))

def argmax(obj, axis=None, skipna=True):
    """ similar to numpy's argmax, but return axis values instead of integer position
    """
    obj, idx, name = _deal_with_axis(obj, axis)
    res = obj.apply('argmax', axis=idx, skipna=skipna)

    # along axis: single axis value
    if axis is not None: # res is Dimarray
	res.values = obj.axes[idx].values[res.values] 
	return res

    # flattened array: tuple of axis values
    else: # res is ndarray
	res = np.unravel_index(res, obj.shape)
	return tuple(obj.axes[i].values[v] for i, v in enumerate(res))

#def _apply_minmax(obj, funcname, axis=None, skipna=True, args=(), **kwargs):
#    """ apply min/max/argmin/argmax
#
#    special behaviour for these functions, with `keepdims` parameter
#    """
#    # get actual axis values instead of numpy's integer index
#    if funcname in ("argmax","argmin","nanargmax","nanargmin"):
#	assert axis is not None, "axis must not be None for "+funcname+", or apply on values"
#	return obj.axes[idx].values[result] 



#
# Define weighted mean/std/var
#

def weighted_mean(obj, axis=None, skipna=True, weights='axis'):
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
	return obj.apply("mean", axis=axis, skipna=skipna)

    # weighted mean
    sum_values = (obj*weights).apply("sum", axis=axis, skipna=skipna)
    sum_weights = weights.apply("sum", axis=axis, skipna=skipna)
    return sum_values / sum_weights

def weighted_var(obj, axis=None, skipna=True, weights="axis", ddof=0):
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
	return obj.apply("var", axis=axis, skipna=skipna, ddof=ddof)

    # weighted mean
    mean = obj.mean(axis=axis, skipna=skipna, weights=weights)
    dev = (obj-mean)**2
    return dev.mean(axis=axis, skipna=skipna, weights=weights)

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
	slice_ = obj.xs(axisval, axis=axis.name, method="index")
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
