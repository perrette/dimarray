""" Module to deal with indexing
"""
import numpy as np
import functools
import copy 

from axes import Axis, Axes, GroupedAxis, is_regular, make_multiindex

#__all__ = ["take", "put", "reindex_axis", "reindex_like"]
__all__ = ["take", "put"]

TOLERANCE=1e-8
MATLAB_LIKE_INDEXING = True

def _get_keyword_indices(obj, indices, axis=0):
    """ get dictionary of indices
    """
    try:
	axes = obj.axes
    except:
	raise TypeError(" must be a DimArray instance !")

    # convert to dictionary
    if type(indices) is tuple or isinstance(indices, dict) or isinstance(indices, Axis):
	assert axis in (None, 0), "cannot have axis > 0 for tuple (multi-dimensional) indexing"

    if isinstance(indices, Axis):
	newaxis = indices
	kw = {newaxis.name: newaxis.values}

    elif type(indices) is tuple:
	kw = {axes[i].name: ind for i, ind in enumerate(indices)}

    elif type(indices) in (list, np.ndarray) or np.isscalar(indices):
	kw = {axes[axis].name:indices}

    elif isinstance(indices, dict):
	kw = indices

    else:
	raise TypeError("indices not understood: {}, axis={}".format(indices, axis))

    return kw

@property
def size(self):
    return np.size(self.values)


def take(obj, indices, axis=0, indexing="values", tol=TOLERANCE, keepdims=False, matlab_like=None):
    """ Retrieve values from a DimArray

    input:

	- self or obj: DimArray (ignore this parameter if accessed as bound method)
	- indices  : int or list or slice (single-dimensional indices)
	             or a tuple of those (multi-dimensional)
		     or `dict` (`axis name` : `indices`)
	- axis     : int or str
	- indexing     : "values" or "position" 
		     "position": use numpy-like position index
		     "values": indexing on axis values 
	- tol	   : tolerance when looking for floats
	- keepdims : keep singleton dimensions
	- matlab_like: if True, indexing with list or array of indices will behave like
	    Matlab TM does, which means that it will index on each individual dimensions.
	    (internally, any list or array of indices will be converted to a boolean index
	    of values before slicing)

	    If False, numpy rules are followed. Consider the following case:

	    a = DimArray(np.zeros((4,4,4)))
	    a[[0,0],[0,0],[0,0]]
	    
	    if matlab_like is True, the result will be a 3-D array of shape 2 x 2 x 2
	    if matlab_like is False, the result will be a 1-D array of size 2. This case 
	    is more complicated to implement because the axes need to be grouped, and it 
	    is ***not yet implemented***. See examples.
	    
	    Note if the "keepdims" options is True, matlab_like is set to True automatically.
	    The default is set at the module level by MATLAB_LIKE_INDEXING, at present
	    default to True, may be set to False (for consistency with numpy) when the feature 
	    with GroupedAxis is implemented.

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

    Indexing via integer index (indexing="position" or `ix` property)
    >>> np.all(v.ix[:,0] == v[:,10])
    True
    >>> np.all(v.take(0, axis="d1", indexing="position") == v.take(10, axis="d1"))
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

    # Matlab like multi-indexing
    >>> v = DimArray(np.arange(2*3*4).reshape(2,3,4))
    >>> v[[0,1],:,[0,0,0]].shape
    (2,3,3)
    >>> v[[0,1],:,[0,0]].shape # here no broadcasting, matlab-like
    (2,3,2)
    >>> v.values[[0,1],:,[0,0]].shape # that is traditional numpy, with broadcasting on same shape
    (2,3)
    """
    assert indexing in ("position", "values"), "invalid mode: "+repr(indexing)

    # 
    # MATLAB-LIKE OR NUMPY-LIKE?
    #
    if matlab_like is None:
	matlab_like = MATLAB_LIKE_INDEXING

    if keepdims is True:
	matlab_like = True

    # matlab-like
    if isinstance(indices, tuple) or isinstance(indices, dict):
	n_int_indices = sum(type(ix) is int for ix in np.index_exp[indices])
	n_array_indices = sum(type(ix) in (list, np.ndarray) for ix in np.index_exp[indices])
	chained_call = n_array_indices > 1 or (n_array_indices == 1 and n_int_indices >=1)
    else:
	chained_call = False

    if chained_call and not matlab_like:
	raise NotImplementedError("advanced indexing only works in matlab mode: no change in dimension")

    # proceed to chained call to avoid weird situations like a[0,:,[0,0]]
    if chained_call:
	if isinstance(indices, tuple):
	    indices = {obj.axes[k].name:ix for k, ix in enumerate(indices)}
	assert isinstance(indices, dict), "pb in chained call"
	for k in indices:
	    obj = take(obj, indices[k], axis=k, keepdims=keepdims, indexing=indexing, tol=tol)
	return obj

    try:
	indices_numpy = obj.axes.loc(indices, axis=axis, position_index=(indexing == "position"), keepdims=keepdims, tol=tol)
    except IndexError, msg:
	raise IndexError(msg)

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


def put(obj, val, indices, axis=0, indexing="values", tol=TOLERANCE, convert=False, inplace=False):
    """ Put new values into DimArray (inplace)

    parameters:
    -----------
    obj: DimArray (do not provide if method bound to class instance)
    val: value to put in: scalar or array-like with appropriate shape
    indices: single- or multi- or `bool` index
    axis: for single index (see help on `take` method and axes Locator)
    indexing : "position", "values"
    convert: convert array to val's type
    inplace: True

    returns:
    --------
    None: (inplace modification)

    Examples:
    ---------

    >>> a = DimArray(np.zeros((2,2)), [('d0',['a','b']), ('d1',[10.,20.])])

    Index by values
    >>> b = a.put(1, indices={'d0': 'b'})
    >>> b
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  0.],
           [ 1.,  1.]])
    >>> a['b'] = 2   # slicing equivalent
    >>> a
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  0.],
           [ 2.,  2.]])

    Index by position
    >>> b = a.put(3, indices=1, axis='d1', indexing="position")
    >>> b
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  3.],
           [ 2.,  3.]])
    >>> a.ix[:,1] = 4  # slicing equivalent
    >>> a
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 2.,  4.]])


    Multi-dimension, multi-index
    >>> b = a.put(5, indices={'d0':'b', 'd1':[10.]})
    >>> b
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 5.,  4.]])
    >>> a["b",[10]] = 6
    >>> a
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 6.,  4.]])

    Inplace
    >>> a.put(6, indices={'d0':'b', 'd1':[10.]}, inplace=True)
    """
    assert indexing in ("position", "values"), "invalid mode: "+repr(indexing)
    indices_numpy = obj.axes.loc(indices, axis=axis, position_index=(indexing == "position"), tol=tol)

    if not inplace:
	obj = obj.copy()

    try:
	obj.values[indices_numpy] = val 

    # convert to val.dtype if needed
    except ValueError, msg:
	if not convert: raise
	dtype = np.asarray(val).dtype
	if obj.dtype is not dtype:
	    obj.values = np.asarray(obj.values, dtype=dtype)
	obj.values[indices_numpy] = val 

    if not inplace:
	return obj

#
# Variants 
#
def take_na(obj, indices, axis=0, indexing="values", tol=TOLERANCE, keepdims=False, na=np.nan, repna=True):
    """ like take but replace any missing value with NaNs

    additional parameters:
    na : replacement value, by default np.nan

    Examples:
    """
    assert indexing in ("position", "values"), "invalid mode: "+repr(indexing)
    indices_numpy = obj.axes.loc(indices, axis=axis, position_index=(indexing == "position"), keepdims=keepdims, tol=tol, raise_error=not repna)

    indices_numpy, indices_mask = _filter_bad_indices(indices_numpy, obj.dims)
    result = take(obj, indices_numpy, indexing="position")
#    if np.isscalar(result):
#	return result if indices_mask is None else na
    put(result, na, indices_mask, convert=True, inplace=True, tol=tol, indexing="position")

    # correct axis values
    if indices_mask is not None:

	#multi_indices = make_multiindex(indices, len(obj.axes)) # tuple of appropriate size
	indices = np.index_exp[indices] # make sure we have a tuple
	for i, dim in enumerate(indices_mask):
	    ix = indices_mask[dim]
	    result.axes[dim].values[ix] = np.asarray(indices[i])[ix]

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
    multi_index = np.index_exp[multi_index]

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
	valid_indices = np.zeros_like(bad, dtype=int)
	valid_indices[~bad] = [indices[k] for k, skip in enumerate(bad) if not skip]

	indices_mask[dim] = bad
	indices_nanfree[i] = valid_indices

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
##
##
### Pick-up values 
##if mode == "interp":
##    result = _take_interp(obj, indices_numpy)
##
##

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

def reindex_axis(self, values, axis=0, method='exact', repna=True, tol=TOLERANCE):
    """ reindex an array along an axis

    Input:
	- values : array-like or Axis: new axis values
	- axis   : axis number or name
	- method : "exact" (default), "nearest", "interp" 
	- repna: if False, raise error when an axis value is not present 
	               otherwise just replace with NaN. Defaulf is True

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
    if method == "exact":
	newobj = take_na(self, values, axis=axis, repna=repna)

    elif method in ("nearest", "interp"):
	from interpolation import interp
	newobj = interp(self, values, axis=axis, method=method, repna=repna)

    else:
	raise ValueError("invalid reindex_axis method: "+repr(method))

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



