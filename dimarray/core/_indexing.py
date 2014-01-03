""" Module to deal with indexing
"""
import numpy as np
import copy 

from axes import Axis, Axes, GroupedAxis

def take(self, ix=None, axis=0, method=None, keepdims=False, raise_error=True, fallback=np.nan):
    """ Retrieve values from a DimArray

    input:

	- ix       : int or list or slice (single-dimensional indices) or a tuple of those (multi-dimensional)
	- axis     : int or str
	- method   : indexing method when 
		     - "numpy": numpy-like integer indexing
		     - "exact": locate based on axis values
		     - "nearest": nearest match, bound checking
		     - "interp": linear interpolation between nearest match and values immediately above.
		     default is "exact"
	- keepdims : keep singleton dimensions

	- raise_error: raise error if index not found? (default True)
	- fallback: replacement value if index not found and raise_error is False (default np.nan)

    output:
	- DimArray object or python built-in type, consistently with numpy slicing

    See also:
    ---------
    take_kw

    Examples:
    ---------

    >>> v = DimArray([[1,2,3],[4,5,6]], axes=[["a","b"], [10.,20.,30.]], dims=['d0','d1'], dtype=float) 
    >>> v
    dimarray: 6 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (3): 10.0 to 30.0
    array([[ 1.,  2.,  3.],
	   [ 4.,  5.,  6.]])

    Indexing via axis values (default, method="exact")
    >>> a = v[:,10]   # python slicing method
    >>> a
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0'
    0 / d0 (2): a to b
    array([ 1.,  4.])
    >>> b = v.take(10, axis=1)  # take, by axis position
    >>> c = v.take(10, axis='d1')  # take, by axis name
    >>> np.all(a == b == c)
    True
    >>> v["a", 10]  # also work with string axis
    1.0

    Indexing via integer index (method="numpy" or `ix` property)

    >>> np.all(v.ix[:,0] == v[:,10])
    True
    >>> np.all(v.take(0, axis="d1", method="numpy") == v.take(10, axis="d1"))
    True

    Experimental: "nearest" and "interp" methods (for int and float axes only)
    >>> v.take(12, axis="d1", method='nearest')
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0'
    0 / d0 (2): a to b
    array([ 1.,  4.])
    >>> v.take(12, axis="d1", method='interp')
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0'
    0 / d0 (2): a to b
    array([ 1.2,  4.2])

    Take a list of indices (default, "exact" method)
    >>> a = v[:,[10,20]] # also work with a list of index
    >>> a
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 1.,  2.],
	   [ 4.,  5.]])
    >>> b = v.take([10,20], axis='d1')
    >>> np.all(a == b)
    True

    Take a slice:
    >>> c = v[:,10:20] # axis values: slice includes last element
    >>> c
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 1.,  2.],
	   [ 4.,  5.]])
    >>> d = v.take(slice(10,20), axis='d1') # `take` accepts `slice` objects
    >>> np.all(c == d)
    True
    >>> v.ix[:,0:1] # integer position: does *not* include last element
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (1): 10.0 to 10.0
    array([[ 1.],
           [ 4.]])

    Multi-dimensional indexing just as []
    >>> a = v["a", 20]
    >>> b = v.take(("a", 20))
    >>> np.all(a == b)
    True

    Keep dimensions 
    >>> a = v[["a"]]
    >>> b = v.take("a",keepdims=True)
    >>> np.all(a == b)
    True
    """
    if method is None:
	method = self._indexing

    # multi-dimensional slicing? 
    if type(ix) is tuple:
	assert axis in (None, 0), "cannot have axis > 0 for tuple (multi-dimensional) indexing"
	indices = [(self.axes[i].name, val) for i, val in enumerate(ix)]
	obj = self
	for name, val in indices:
	    obj = take(obj, val, axis=name, method=method, keepdims=keepdims, raise_error=raise_error, fallback=fallback)
	return obj

    assert axis is not None, "axis= must be provided"
    assert type(ix) is not tuple, "axis is tuple, pb with multidimensional slicing"

    axis, name = self._get_axis_info(axis)

    # get an axis object
    ax = self.axes[axis] # axis object

    # numpy-like indexing, do nothing
    if method in ("numpy"):
	#if type(ix) is tuple: ix = slice(*ix)
	indices = ix

    # otherwise locate the values
    elif method in ('exact','nearest', 'interp'):
	indices = ax.loc(ix, method=method, raise_error=raise_error, fallback=fallback) 

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
	result = _take_interp(self, indices, axis=axis, keepdims=keepdims)
	# fix type
	ix = np.array(ix) # make sure ix is an array
	if axis < result.ndim:
	    result.axes[axis].values = np.array(result.axes[axis].values, dtype=ix.dtype)

    else:
	result = _take_numpy(self, indices, axis=axis, keepdims=keepdims)

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


def take_kw(self, *opts, **kwaxes):
    """ A version of take that accept keyword arguments for slicing (EXPERIMENTAL)

    *opt: options as unnamed, variable-list arguments.
	   ["method", ["keepdims", ["raise_error", ["fallback"]]]]: see take for explanation
    **kwaxes: `axis name = axis value(s)` 

    See also:
    ---------
    take

    Examples:
    ---------
    >>> v = DimArray([[1,2,3],[4,5,6]], axes=[["a","b"], [10.,20.,30.]], dims=['d0','d1'], dtype=float) 
    >>> a = v.take_kw(d1=20)  # take, as keyword argument: EXPERIMENTAL
    >>> b = v.take_kw("numpy", d1=1)  # add "method" as first argument
    >>> c = v.take(20, axis="d1")
    >>> np.all(a == b == c)
    True
    """
    # get options from variable list
    list_of_opts = ["method", "keepdims","raise_error", "fallback"]
    kwopt = {}
    for i, o in enumerate(opts):
	kwopt[list_of_opts[i]] = o
	 
    # a chained call: multi-dimensional slicing <axis name> : <axis index value>
    obj = self
    for nm, idx in kwaxes.iteritems():
	obj = take(obj, idx, axis=nm, **kwopt)

    return obj



def put(obj, ix, axis=0, method=None):
    """ Put new values into DimArray
    """
    if method is None:
	method = self._indexing
     

# default axis = None !!

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

    # If result is a numpy array, make it a DimArray
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
    v0 = take(obj, i0, axis=axis, method="numpy", keepdims=keepdims)
    v1 = take(obj, i1, axis=axis, method="numpy", keepdims=keepdims)

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

def reindex_axis(self, values, axis=0, method='exact', raise_error=False):
    """ reindex an array along an axis

    Input:
	- values : array-like or Axis: new axis values
	- axis   : axis number or name
	- method : "exact" (default), "nearest", "interp" (see take)
	- raise_error: if True, raise error when an axis value is not present 
	               otherwise just fill-in with NaN. Defaulf is False.

    Output:
	- DimArray

    Examples:
    ---------

    Basic reindexing: fill missing values with NaN
    >>> a = da.DimArray([1,2,3],('x0', [1,2,3]))
    >>> b = da.DimArray([3,4],('x0',[1,3]))
    >>> b.reindex_axis([1,2,3])
    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 1 to 3
    array([  3.,  nan,   4.])

    "nearest" mode
    >>> b.reindex_axis([0,1,2,3], method='nearest') # out-of-bound to NaN
    dimarray: 3 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (4): 0 to 3
    array([ nan,   3.,   3.,   4.])

    "interp" mode
    >>> b.reindex_axis([0,1,2,3], method='interp') # out-of-bound to NaN
    dimarray: 3 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (4): 0 to 3
    array([ nan,  3. ,  3.5,  4. ])
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

    # new DimArray
    # ...replace the axis
    ax0 = Axis(values, ax.name)
    ax1 = newobj.axes[axis_id]
    
    #assert np.all((np.isnan(ax0.values) | (ax0.values == ax1.values))), "pb when reindexing"

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

    Example:
    --------

    >>> b = da.DimArray([3,4],('x0',[1,3]))
    >>> c = da.DimArray([[1,2,3], [1,2,3]],[('x1',["a","b"]),('x0',[1, 2, 3])])
    >>> b.reindex_like(c, method='interp')
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 1 to 3
    array([ 3. ,  3.5,  4. ])
    """
    return _reindex_axes(self, other.axes, method=method, **kwargs)
