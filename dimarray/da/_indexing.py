""" Module to deal with indexing
"""
import numpy as np
import copy 

from axes import Axis, Axes, GroupedAxis, Locator_axis, Locator_axes

def take(self, ix=None, axis=0, method=None, keepdims=False, raise_error=True, missing=np.nan):
    """ Retrieve values from a DimArray

    input:

	- ix       : int or list or slice (single-dimensional indices)
	             or a tuple of those (multi-dimensional)
		     or `dict` (`axis name` : `indices`)
	- axis     : int or str
	- method   : indexing method when 
		     - "numpy": numpy-like integer indexing
		     - "exact": locate based on axis values
		     - "nearest": nearest match, bound checking
		     - "interp": linear interpolation between nearest match and values immediately above.
		     default is "exact"
	- keepdims : keep singleton dimensions

	- raise_error: raise error if index not found? (default True)
	- missing: replacement value if index not found and raise_error is False (default np.nan)

    output:
	- DimArray object or python built-in type, consistently with numpy slicing

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
    >>> d = v.take({'d1':10})  # take, by dict {axis name : axis values}
    >>> np.all(a == b == c == d)
    True

    Indexing via integer index (method="numpy" or `ix` property)
    >>> np.all(v.ix[:,0] == v[:,10])
    True
    >>> np.all(v.take(0, axis="d1", method="numpy") == v.take(10, axis="d1"))
    True

    Multi-dimensional indexing
    >>> v["a", 10]  # also work with string axis
    1.0
    >>> v.take(('a',10))  # multi-dimensional, tuple
    1.0
    >>> v.take({'x0':'a', 'x1':10})  # multi-dimensional, keyword args
    1.0

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

    Keep dimensions 
    >>> a = v[["a"]]
    >>> b = v.take("a",keepdims=True)
    >>> np.all(a == b)
    True
    """
    if method is None:
	method = self._indexing
    #assert axis is not None, "axis= must be provided"
    #axis, name = self._get_axis_info(axis)

    return _take_check(self, ix=ix, axis=axis, method=method, keepdims=keepdims, raise_error=raise_error, missing=missing)

def put(obj, val, ix, axis=0, method=None):
    """ Put new values into DimArray (inplace)

    parameters:
    -----------
    obj: DimArray (do not provide if method bound to class instance)
    val: value to put in: scalar or array-like with appropriate shape
    ix: single- or multi- or `bool` index
    axis: for single index (see help on `take` method and axes Locator)
    method: see `take`

    returns:
    --------
    None: (inplace modification)

    Examples:
    ---------
    """
    if method is None:
	method = obj._indexing

    # locate indices
    multi_index = Locator_axes(obj.axes, method=method)(ix, axis=axis)
    obj.values[multi_index] = val 

def _take(obj, indices, **kwargs):
    """ take indices from one of the standard methods ("numpy", "exact", "nearest")
    """
    # Convert to a tuple numpy indices matching the shape of the DimArray
    indices_numpy = Locator_axes(indices, **kwargs)

    # New values
    newval = obj.values[indices_numpy] # new numpy values

    # New axes
    newaxes = Axes()
    stamps = []
    for i, ix in enumerate(indices_numpy)
	ax = obj.axes[i]
	newaxis = ax[ix]
	if np.isscalar(newaxis):
	    stamps += ["{}={}".format(ax.name, newaxis)]
	else:
	    newaxes.append(newaxis)
    stamp = ",".join(stamps)

    # If result is a numpy array, make it a DimArray
    if isinstance(newval, np.ndarray):
	result = obj._constructor(newval, axes, **obj._metadata)

	# append stamp
	if stamp: result._metadata_stamp(stamp)

    # otherwise, return scalar
    else:
	result = newval

    return result


def _take_check(obj, indices, method="exact", **kwargs):
    """ same as take but also nans indices
    """
    # Convert to a tuple numpy indices matching the shape of the DimArray
    indices_numpy = Locator_axes(obj.axes, **kwargs)(indices)

    # Filter bad from good indices: this would happen only for raise_error == False
    indices_numpy, bad_ix = _get_bad_indices(indices_numpy)

    # Pick-up values 
    if method == "interp":
	result = _take_interp(obj, indices_numpy)
	#    # fix type
	#    ix = np.asarray(ix) # make sure ix is an array
	#    if not np.isscalar(result):
	#	result.axes[axis].values = np.array(result.axes[axis].values, dtype=ix.dtype)

    else:
	result = _take(obj, indices_numpy, method="numpy")

    if bad_ix is not None:
	result = _fill_bad_indices(result, bad_ix, indices)

    return result

def _get_bad_indices(indices_numpy):
    """ replace NaN indices by zeros

    input:
	indices_numpy: tuple of `int` or `slice` objects
    output:
	indices_nanfree: same as indices_numpy with NaNs replaced with 0
	bad_ix: `dict` of `bool` 1-D arrays to indicate the locations of bad indices
    """
    bads = {}
    indices_nanfree = copy.copy(indices_numpy)
    for i, ind in enumerate(indices_numpy):
	if type(ind) is slice: 
	    continue
	bad = np.isnan(ind)
	if np.size(bad) > 0:
	    ind = np.where(bad,0,ind)
	    if not method == "interp":
		ind = np.array(ind, dtype=int) # NaNs produced conversion to float
	    bads[i] = bad # record bad indices
	indices_nanfree[i] = ind

    return indices_nanfree, bads

def _fill_bad_indices(result, bad_ix, indices, missing=np.nan):
    """ fill NaN back in

    input: 
	result: DimArray
	bad_ix: `dict` of `bool` 1-D arrays to indicate the locations of bad indices
	indices: originally required indices (n-d)
	missing: replacement values for bad indices (default NaNs)

    output:
	result: corrected DimArray with bad values replaced with `missing` 
    """
    for k in bad_ix:
	bad = bad_ix[k] # `bool` indices of bad numbers

	# replace with originally asked-for values
	result.axes[k].values[bad] = indices[bad] 

	# replace array values with NaNs 
	pos = result.dims.index(k) # index of the new axis position
	ix = pos*(slice(None),) + bad, # multi-index of bad positions

	# convert to float 
	if result.dtype is not np.dtype(missing):
	    result.values = np.asarray(result.values, dtype=np.dtype(missing))
	result.values[ix] = missing
	#result.put(missing, bad_ix[k], axis=k)


def _take_interp(obj, indices):
    """ multi-indices
    """
    indices = np.index_exp[indices]
    dims = [obj.dims[i] for i in range(len(indices))]
    for i, ix in enumerate(indices):
	obj = _take_interp_axis(obj, ix, axis=dims[i])
    return obj

def _take_interp_axis(obj, indices, axis):
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
    v0 = take(obj, i0, axis=axis, method="numpy")
    v1 = take(obj, i1, axis=axis, method="numpy")

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
