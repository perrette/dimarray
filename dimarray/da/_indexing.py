""" Module to deal with indexing
"""
import numpy as np
import functools
import copy 

from axes import Axis, Axes, GroupedAxis, is_regular, make_multiindex

#__all__ = ["take", "put", "reindex_axis", "reindex_like"]
__all__ = ["take", "put"]


@property
def size(self):
    return np.size(self.values)


def take(obj, indices, axis=0, mode=None, tol=1e-8, keepdims=False):
    """ Retrieve values from a DimArray

    input:

	- self or obj: DimArray (ignore this parameter if accessed as bound method)
	- indices  : int or list or slice (single-dimensional indices)
	             or a tuple of those (multi-dimensional)
		     or `dict` (`axis name` : `indices`)
	- axis     : int or str
	- mode     : None or "values" or "numpy" 
		     "numpy": use numpy-like position index instead of values-index
		     "values": indexing on axis values 
		     The default (None) is to look at obj._indexing_mode
	- tol	   : tolerance when looking for floats, default 1e-8
	- keepdims : keep singleton dimensions

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

    Indexing via axis values (default)
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

    Indexing via integer index (mode="numpy" or `ix` property)
    >>> np.all(v.ix[:,0] == v[:,10])
    True
    >>> np.all(v.take(0, axis="d1", mode="numpy") == v.take(10, axis="d1"))
    True

    Multi-dimensional indexing
    >>> v["a", 10]  # also work with string axis
    1.0
    >>> v.take(('a',10))  # multi-dimensional, tuple
    1.0
    >>> v.take({'d0':'a', 'd1':10})  # dict-like arguments
    1.0

    Take a list of indices
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

    tolerance parameter to achieve "nearest neighbour" search
    >>> v.take(12, axis="d1", tol=5)
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0'
    0 / d0 (2): a to b
    array([ 1.,  4.])

    """
    if mode is None: mode = obj._indexing_mode

#    if mode == "interp":
#	return interp(obj, indices, axis=axis, tol=tol, keepdims=keepdims)
#
#    elif mode == "nearest":
#	return take_nearest(obj, indices, axis=axis, tol=tol, keepdims=keepdims)

    if mode is None: mode = "values"

    # Convert to a tuple numpy indices matching the shape of the DimArray
    assert mode in ("index", "values"), "invalid mode: "+repr(mode)
    indices_numpy = obj.axes.loc(indices, axis=axis, position_index=(mode is "numpy"), keepdims=keepdims, tol=tol)

    # New values
    newval = obj.values[indices_numpy] # new numpy values

    # New axes
    newaxes = Axes()
    stamps = []
    for i, ix in enumerate(indices_numpy):
	ax = obj.axes[i]
	newaxis = ax[ix]
	if np.isscalar(newaxis):
	    stamps += ["{}={}".format(ax.name, newaxis)]
	else:
	    newaxes.append(newaxis)
    stamp = ",".join(stamps)

    # If result is a numpy array, make it a DimArray
    if isinstance(newval, np.ndarray):
	result = obj._constructor(newval, newaxes, **obj._metadata)

	# append stamp
	if stamp: result._metadata_stamp(stamp)

    # otherwise, return scalar
    else:
	result = newval

    return result


def put(obj, val, indices, axis=0, position_index=None, convert=False):
    """ Put new values into DimArray (inplace)

    parameters:
    -----------
    obj: DimArray (do not provide if method bound to class instance)
    val: value to put in: scalar or array-like with appropriate shape
    indices: single- or multi- or `bool` index
    axis: for single index (see help on `take` method and axes Locator)
    position_index : if True, consider indices as numpy-like position indices
    convert: convert array to val's type

    returns:
    --------
    None: (inplace modification)

    Examples:
    ---------

    >>> a = DimArray(np.zeros((2,2)), [('d0',['a','b']), ('d1',[10.,20.])])

    By axis values:
    >>> a.put(1, indices='b', axis='d0')
    >>> a
    >>> a['b'] = 2   # slicing equivalent
    >>> a

    By position index:
    >>> a.put(3, indices=1, axis='d0', position_index=True)
    >>> a
    >>> a.ix[:,1] = 4  # slicing equivalent
    >>> a

    Multi-dimension, multi-index
    >>> a.put(5, indices={'d0':'b', 'd1':[10.]})
    >>> a
    >>> a["b",[10]] = 6
    >>> a
    """
    if position_index is None:
	position_index = obj._numpy_indexing

    # locate indices
    multi_index = obj.axes.loc(position_index=position_index)(indices, axis=axis)

    # convert to val.dtype if needed
    if convert:
	dtype = np.asarray(val).dtype
	if obj.dtype is not dtype:
	    obj.values = np.asarray(obj.values, dtype=dtype)

    obj.values[multi_index] = val 

    return obj

#
# Variants 
#
def take_na(obj, indices, axis=0, position_index=False, tol=1e-8, keepdims=False, na=np.nan):
    """ like take but replace any missing value with NaNs

    additional parameters:
    na : replacement value, by default np.nan

    Examples:
    """
    indices_numpy = obj.axes.loc(raise_error=False, position_index=position_index, keepdims=keepdims, tol=tol)(indices, axis=axis)
    indices_numpy, indices_mask = _filter_bad_indices(indices_numpy, obj.dims)
    result = take(obj, indices_numpy, position_index = True)
    result.put(np.nan, indices_mask, convert=True)

    # correct axis values
    if indices_mask is not None:

	#multi_indices = make_multiindex(indices, len(obj.axes)) # tuple of appropriate size
	indices = np.index_exp[indices] # make sure we have a tuple
	for i, dim in enumerate(indices_mask):
	    ix = indices_mask[dim]
	    result.axes[dim].values[ix] = indices[i][ix]

    return result

def _filter_bad_indices(multi_index, dims):
    """ replace NaN indices by zeros

    input:
	multi_index: tuple of `int` or `slice` objects
    output:
	indices_nanfree: same as multi_index with NaNs replaced with 0
	indices_mask: multi-index of NaNs for the sliced array
	bad_ix: `dict` of `bool` 1-D arrays to indicate the locations of bad indices
    """
    assert type(multi_index) is tuple

    indices_nanfree = list(multi_index)
    indices_mask = {}

    for i, indices in enumerate(multi_index):
	dim = dims[i]

	if indices is None: 
	    indices_mask[dim] = slice(None) # all NaNs

	if type(indices) is slice: 
	    continue

	if type(indices) is int: 
	    continue

	assert np.iterable(indices)

	bad = np.array([ix is None for ix in indices])
	indices = np.where(bad, np.zeros_like(bad), indices)

	indices_mask[dim] = bad
	indices_nanfree[i] = indices

    return tuple(indices_nanfree), indices_mask

###
### INTERP MODE?
###
##
##    # Pick-up values 
##    if mode == "interp":
##	result = _take_interp(obj, indices_numpy)
##
##
##def interp(self, val):
##    """ return fractional index for interpolation
##    """
##    assert is_regular(self.values), "interp mode only makes sense for regular axes !"
##
##    # to handle nan data, otherwise pb with values[i]
##    if np.iterable(val):
##	return [self.interp(v) for v in val]
##
##    # index of nearest neighbour
##    i = self.locate(val, mode="nearest")
##
##    if np.isnan(i): return np.nan
##
##    xi = self.values[i]
##
##    # axis step
##    dx = float(self.values[1] - self.values[0])
##    return i + (val-xi)/dx
##
##
### Pick-up values 
##if mode == "interp":
##    result = _take_interp(obj, indices_numpy)
##
##
##def _take_interp(obj, indices):
##    indices = np.index_exp[indices]
##    kw = {obj.axes[i].name:ix for i, ix in enumerate(indices)}
##    for k in kw:
##	if type(kw[k]) is slice:
##	    obj = take(obj, kw[k], k, mode="numpy")
##	else:
##	    obj = _take_interp_axis(obj, kw[k], k)
##    return obj
##
##def _take_interp_axis(obj, indices, axis):
##    """ Take a number or an integer as from an axis
##    """
##    assert type(indices) is not slice, "interp only work with integer and list indexing"
##
##    # return a "fractional" index
##    ax = obj.axes[axis]
##
##    # position indices of nearest neighbors
##    # ...last element, 
##    i0 = np.array(indices, dtype=int)
##    i1 = i0+1
##
##    # make sure we are not beyond 
##    over = i1 == ax.size
##    if i1.size == 1:
##	if over:
##	    i0 -= 1
##	    i1 -= 1
##    else:
##	i0[over] = ax.size-2
##	i1[over] = ax.size-1
##
##    # weight for interpolation
##    w1 = indices-i0 
##
##    # sample nearest neighbors
##    v0 = take(obj, i0, axis=axis, mode="numpy")
##    v1 = take(obj, i1, axis=axis, mode="numpy")
##
##    # result as weighted sum
##    if not hasattr(v0, 'values'): # scalar
##	return v0*(1-w1) + v1*w1
##
###    values = 0
###    for i, w in enumerate(w1):
###	i_nd = (slice(None),)*axis + (i,)
###	values += v0.values[i_nd]*(1-w) + v1.values[i_nd]*w
##    values = v0.values*(1-w1) + v1.values*w1
##
##    axes = []
##    for d in v0.dims:
##
##	# only for list indexing or if keepdims is True
##	if d == ax.name:
##	    axis = Axis(v0.axes[d].values*(1-w1) + v1.axes[d].values*w1, d)
##
##	# other dimensions not affected by slicing
##	else:
##	    axis = obj.axes[d]
##	axes.append(axis)
##    return obj._constructor(values, axes, **obj._metadata)


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

##    if position_index: repna = False # do not fill in with NaNs if position index
##
##    # Filter bad from good indices: this would happen only for repna=True
##    if repna:
##	indices_numpy, indices_mask = _filter_bad_indices(indices_numpy)
##
##    #
##    # SLICE or PICK ITEMS
##    #
##	
##
##    #
##    # NOW REPLACE MISSING DATA with NaNs
##    #
##    if repna and indices_mask is not None:
##	result = _fill_bad_indices(result, indices_mask, indices)

#    # fix type in interp mode
#    ix = np.asarray(ix) # make sure ix is an array
#    if not np.isscalar(result):
#	result.axes[axis].values = np.array(result.axes[axis].values, dtype=ix.dtype)


## def reindex_axis(self, values, axis=0, method='exact', repna=True):
##     """ reindex an array along an axis
## 
##     Input:
## 	- values : array-like or Axis: new axis values
## 	- axis   : axis number or name
## 	- method : "exact" (default), "nearest", "interp" (see take)
## 	- repna: if False, raise error when an axis value is not present 
## 	               otherwise just replace with NaN. Defaulf is True
## 
##     Output:
## 	- DimArray
## 
##     Examples:
##     ---------
## 
##     Basic reindexing: fill missing values with NaN
##     >>> a = da.DimArray([1,2,3],('x0', [1,2,3]))
##     >>> b = da.DimArray([3,4],('x0',[1,3]))
##     >>> b.reindex_axis([1,2,3])
##     dimarray: 2 non-null elements (1 null)
##     dimensions: 'x0'
##     0 / x0 (3): 1 to 3
##     array([  3.,  nan,   4.])

##     Replace missing values with NaNs
##     >>> v.take([12,20], axis="d1", repna=True)
##     dimarray: 2 non-null elements (0 null)
##     dimensions: 'd0'
##     0 / d0 (2): a to b
##     array([[nan,  2.],
## 	   [nan,  5.]])
## 
##     "nearest" mode
##     >>> b.reindex_axis([0,1,2,3], method='nearest') # out-of-bound to NaN
##     dimarray: 3 non-null elements (1 null)
##     dimensions: 'x0'
##     0 / x0 (4): 0 to 3
##     array([ nan,   3.,   3.,   4.])
## 
##     "interp" mode
##     >>> b.reindex_axis([0,1,2,3], method='interp') # out-of-bound to NaN
##     dimarray: 3 non-null elements (1 null)
##     dimensions: 'x0'
##     0 / x0 (4): 0 to 3
##     array([ nan,  3. ,  3.5,  4. ])
##     """
##     if isinstance(values, Axis):
## 	newaxis = values
## 	values = newaxis.values
## 	axis = newaxis.name
## 
##     axis_id = self.axes.get_idx(axis)
##     axis_nm = self.axes.get_idx(axis)
##     ax = self.axes[axis_id] # Axis object
## 
##     # do nothing if axis is same or only None element
##     if ax.values[0] is None or np.all(values==ax.values):
## 	return self
## 
##     # indices along which to sample
##     if method in ("nearest","exact","interp", None):
## 	newobj = self.take(values, axis=axis, mode=method, repna=repna)
## 
##     else:
## 	raise ValueError("invalid reindex_axis method: "+repr(method))
## 
##     # new DimArray
##     # ...replace the axis
##     ax0 = Axis(values, ax.name)
##     ax1 = newobj.axes[axis_id]
##     
##     #assert np.all((np.isnan(ax0.values) | (ax0.values == ax1.values))), "pb when reindexing"
## 
##     return newobj
## 
## 
## def _reindex_axes(self, axes, method=None, **kwargs):
##     """ reindex according to a list of axes
##     """
##     obj = self
##     newdims = [ax2.name for ax2 in axes]
##     for ax in self.axes:
## 	if ax.name in newdims:
## 	    newaxis = axes[ax.name].values
## 	    obj = obj.reindex_axis(newaxis, axis=ax.name, method=method, **kwargs)
## 
##     return obj
## 
## def reindex_like(self, other, method=None, **kwargs):
##     """ reindex like another axis
## 
##     note: only reindex axes which are present in other
## 
##     Example:
##     --------
## 
##     >>> b = da.DimArray([3,4],('x0',[1,3]))
##     >>> c = da.DimArray([[1,2,3], [1,2,3]],[('x1',["a","b"]),('x0',[1, 2, 3])])
##     >>> b.reindex_like(c, method='interp')
##     dimarray: 3 non-null elements (0 null)
##     dimensions: 'x0'
##     0 / x0 (3): 1 to 3
##     array([ 3. ,  3.5,  4. ])
##     """
##     return _reindex_axes(self, other.axes, method=method, **kwargs)
## 
## 

