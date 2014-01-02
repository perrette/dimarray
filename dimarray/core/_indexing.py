""" Module to deal with indexing
"""
import numpy as np
import copy 

from axes import Axis, Axes, GroupedAxis

def take(self, ix=None, axis=None, method=None, keepdims=False, raise_error=True, fallback=np.nan, **axes):
    """ Cross-section, can be multidimensional

    input:

	- ix       : int or list or tuple or slice (indices) 
		     if axis is None, does not operate on the flatten array but except a tuple of length `ndim`
	- axis     : int or str or None
	- method   : indexing method when 
		     - "numpy": numpy-like integer indexing
		     - "exact": locate based on axis values
		     - "nearest": nearest match, bound checking
		     - "interp": linear interpolation between nearest match and values immediately above.
		     default is "exact"
	- keepdims : keep singleton dimensions

	- raise_error: raise error if index not found? (default True)
	- fallback: replacement value if index not found and raise_error is False (default np.nan)

	- **axes  : provide axes as keyword arguments for multi-dimensional slicing
		    ==> chained call to take 
		    Note in this mode axis cannot be named after named keyword arguments
		    (`ix`, `axis`, `method` or `keepdims`)

    output:
	- Dimarray object or python built-in type, consistently with numpy slicing

    >>> a.take(45.5, axis=0)	 # doctest: +ELLIPSIS
    >>> a.take(45.7, axis="lat") == a.take(45.5, axis=0) # "nearest" matching
    True
    >>> a.take(time=1952.5)
    >>> a.take(time=70, method="numpy") # 70th element along the time dimension

    >>> a.take(lon=(30.5, 60.5), lat=45.5) == a[:, 45.5, 30.5:60.5] # multi-indexing, slice...
    True
    >>> a.take(time=1952, lon=-40, lat=70, method="nearest") # lookup nearest element (with bound checking)
    """
    if method is None:
	method = self._indexing

    # single-axis slicing
    if ix is not None:
	obj = _take_axis(self, ix, axis=axis, method=method, keepdims=keepdims, raise_error=raise_error, fallback=fallback)

    # multi-dimensional slicing <axis name> : <axis index value>
    # just a chained call
    else:
	obj = self
	for nm, idx in axes.iteritems():
	    obj = _take_axis(obj, idx, axis=nm, method=method, keepdims=keepdims, raise_error=raise_error, fallback=fallback)

    return obj

def put(obj, ix, axis=0, method=None):
    """ Put new values into Dimarray
    """
    if method is None:
	method = self._indexing
     

# default axis = None !!

def _take_axis(obj, ix, axis=0, method="exact", keepdims=False, raise_error=True, **kwargs):
    """ cross-section or slice along a single axis, see take
    """
    assert axis is not None, "axis= must be provided"

    axis, name = obj._get_axis_info(axis)

    # get integer index/slice for axis valued index/slice
    if method is None:
	method = obj._indexing # slicing method

    # get an axis object
    ax = obj.axes[axis] # axis object

    # numpy-like indexing, do nothing
    if method in ("numpy"):
	indices = ix

    # otherwise locate the values
    elif method in ('exact','nearest', 'interp'):
	indices = ax.loc(ix, method=method, raise_error=raise_error, **kwargs) 

    else:
	raise ValueError("Unknown method: "+method)

    # Filter bad from good indices: this would happen only for raise_error == False
    if not raise_error:
	bad = np.isnan(indices)
	if np.size(bad) > 0:
	    indices = np.where(bad,0,indices)
	    if method != "interp":
		indices = np.array(indices, dtype=int) # NaNs produced conversion to float

    # Pick-up values 
    if method == "interp":
	result = _take_interp(obj, indices, axis=axis, keepdims=keepdims)
	# fix type
	ix = np.array(ix) # make sure ix is an array
	if axis < result.ndim:
	    result.axes[axis].values = np.array(result.axes[axis].values, dtype=ix.dtype)

    else:
	result = _take_numpy(obj, indices, axis=axis, keepdims=keepdims)

    # Refill NaNs back in
    if not raise_error and np.size(bad) > 0:

	# make sure we do not have a slice 
	if type(ix) in (slice, tuple):
	    raise ValueError("problem when retrieving value, set raise_error to True for more info")

	ix = np.array(ix) # make sure ix is an array

	bad_nd = (slice(None),)*axis + (bad,)
	result.axes[axis].values[bad] = ix[bad]

	# convert to float 
	if result.dtype is np.dtype(int):
	    result.values = np.array(result.values, dtype=float)
	result.values[bad_nd] = np.nan

    return result


def _take_numpy(obj, indices, axis, keepdims=False):
    """ same as _take, but just for numpy indices
    """
    ax = obj.axes[axis]

    # make a numpy index  and use numpy's slice method (`slice(None)` :: `:`)
    indices_nd = (slice(None),)*axis + (indices,)
    newval = obj.values[indices_nd]
    newaxis = ax[indices] # returns an Axis object

    # if resulting dimension has reduced, remove the corresponding axis
    axes = copy.copy(obj.axes)

    # check for collapsed axis
    collapsed = not isinstance(newaxis, Axis)
	
    # re-expand things even if the axis collapsed
    if collapsed and keepdims:

	newaxis = Axis([newaxis], ax.name) 
	reduced_shape = list(obj.shape)
	reduced_shape[axis] = 1 # reduce to one
	newval = np.reshape(newval, reduced_shape)

	collapsed = False # set as not collapsed

    # If collapsed axis, just remove it and add new stamp
    if collapsed:
	axes.remove(ax)
	stamp = "{}={}".format(ax.name, newaxis)

    # Otherwise just update the axis
    else:
	axes[axis] = newaxis
	stamp = None

    # If result is a numpy array, make it a Dimarray
    if isinstance(newval, np.ndarray):
	result = obj._constructor(newval, axes, **obj._metadata)

	# append stamp
	if stamp: result._metadata_stamp(stamp)

    # otherwise, return scalar
    else:
	result = newval

    return result

def _take_interp(obj, indices, axis, keepdims):
    """ Take a number or an integer as from an axis
    """
    assert type(indices) is not slice, "interp only work with integer and list indexing"

    # return a "fractional" index
    ax = obj.axes[axis]

    # position indices of nearest neighbors
    # ...last element, 
    i0 = np.array(indices, dtype=int)
    i1 = i0+1

    # make sure we are not beyond 
    over = i1 == ax.size
    if i1.size == 1:
	if over:
	    i0 -= 1
	    i1 -= 1
    else:
	i0[over] = ax.size-2
	i1[over] = ax.size-1

    # weight for interpolation
    w1 = indices-i0 

    # sample nearest neighbors
    v0 = _take_axis(obj, i0, axis=axis, method="numpy", keepdims=keepdims)
    v1 = _take_axis(obj, i1, axis=axis, method="numpy", keepdims=keepdims)

    # result as weighted sum
    if not hasattr(v0, 'values'): # scalar
	return v0*(1-w1) + v1*w1

#    values = 0
#    for i, w in enumerate(w1):
#	i_nd = (slice(None),)*axis + (i,)
#	values += v0.values[i_nd]*(1-w) + v1.values[i_nd]*w
    values = v0.values*(1-w1) + v1.values*w1

    axes = []
    for d in v0.dims:

	# only for list indexing or if keepdims is True
	if d == ax.name:
	    axis = Axis(v0.axes[d].values*(1-w1) + v1.axes[d].values*w1, d)

	# other dimensions not affected by slicing
	else:
	    axis = obj.axes[d]
	axes.append(axis)
    return obj._constructor(values, axes, **obj._metadata)


#    #
#    # add an extra "where" method
#    #
#    def where(self, condition, otherwise=None, axis=None):
#	""" 
#	parameters:
#	-----------
#
#	    condition: bool array of same size as self, unless `axis=` is provided
#		 OR    `str` indicating a condition on axes
#	    otherwise: array of same size as self or scalar, replacement value when condition is False
#	    axis     : if provided, interpret the condition as applying along an axis
#
#	returns:
#	--------
#	    
#	    array with self values when condition is True, and `otherwise` if False
#	    if only `condition` is provided, return axis values for which `condition` is True
#
#	Examples:
#	---------
#	    a.where(a > 0)
#	"""
#	# convert scalar to the right shape
#	if np.size(otherwise) == 1:
#	    otherwise += np.zeros_like(self.values)
#
#	# evaluate str condition
#	if type(condition) is str:
#	    result = eval(condition, {ax.name:ax.values})
#
#	result = np.where(condition, [self.values, otherwise])

#
# Reindex axis
#

def reindex_axis(self, values, axis=0, method='values', raise_error=False):
    """ reindex an array along an axis

    Input:
	- values : array-like or Axis: new axis values
	- axis   : axis number or name
	- method : "exact", "nearest", "interp" (see take)
	- raise_error: if True, raise error when an axis value is not present 
	               otherwise just fill-in with NaN. Defaulf is False.

    Output:
	- Dimarray
    """
    if isinstance(values, Axis):
	newaxis = values
	values = newaxis.values
	axis = newaxis.name

    axis_id = self.axes.get_idx(axis)
    axis_nm = self.axes.get_idx(axis)
    ax = self.axes[axis_id] # Axis object

    # do nothing if axis is same or only None element
    if ax.values[0] is None or np.all(values==ax.values):
	return self

    # indices along which to sample
    if method in ("nearest","exact","interp", None):
	newobj = self.take(values, axis=axis, method=method, raise_error=raise_error)

    else:
	raise ValueError("invalid reindex_axis method: "+repr(method))

    # new Dimarray
    # ...replace the axis
    ax0 = Axis(values, ax.name)
    ax1 = newobj.axes[axis_id]
    
    assert np.all((np.isnan(ax0.values) | (ax0.values == ax1.values))), "pb when reindexing"

    return newobj


def _reindex_axes(self, axes, method=None, **kwargs):
    """ reindex according to a list of axes
    """
    obj = self
    newdims = [ax2.name for ax2 in axes]
    for ax in self.axes:
	if ax.name in newdims:
	    newaxis = axes[ax.name].values
	    obj = obj.reindex_axis(newaxis, axis=ax.name, method=method, **kwargs)

    return obj

def reindex_like(self, other, method=None, **kwargs):
    """ reindex like another axis

    note: only reindex axes which are present in other
    """
    return _reindex_axes(self, other.axes, method=method, **kwargs)
