""" Module to deal with indexing
"""
import numpy as np
import copy 

from axes import Axis, Axes, GroupedAxis, is_regular

__all__ = ["take", "put", "Locator_axis", "Locator_axes", "reindex_axis", "reindex_like"]


#
# Locate elements on an axis of values, for different modes
#
def locate_num(values, val, tol=1e-8, mode="raise", modulo=None, regular=None):
    """ locate values in a numeric ndarray

    parameters:
    -----------
    values: ndarray of int or float
    val   : value to locate (`float` or `int`)

    optional parameters:
    --------------------
    mode: different modes to handle out-of-mode situations
	"raise": raise error
	"clip" : returns 0 or -1 (first or last element)
	"wrap" : equivalent to modulo=values.ptp()
    tol: tolerance to find data 
    modulo: val = val +/- modulo*n, where n is an integer (default None)

    output:
    -------
    loc: integer position of val on values

    Examples:
    ---------

    >>> values = [-4.,-2.,0.,2.,4.]
    >>> locate_num(values, 2.)
    3
    >>> locate_num(values, 6, modulo=8)
    1
    >>> locate_num(values, 6, mode="wrap")
    4
    """
    values = np.asarray(values)
    #assert values.dtype in (np.dtype(int), np.ndtype(float)), "numeric array only"
    mi, ma = values.ptp() # min, max

    # modulo calculation, val = val +/- modulo*n, where n is an integer
    if modulo and val < mi or val > ma:
	val = _adjust_modulo(val, modulo, mi)

    if mode:
	if val < mi or val > ma:

	    # check if axis if regular, otherwise only "raise" valid
	    if mode != "raise":
		if regular is None:
		    regular = is_regular(values)
		if not regular:
		    warnings.warning("%s mode only valid for regular axes" % (mode))
		    mode = "raise"

	    if mode == "raise"
		raise ValueError("%f out of bounds ! (min: %f, max: %f)" % (val, mi, ma))

	    elif mode == "clip":
		if val < mi: return 0
		else: return -1

	    elif mode == "wrap":
		span = values[-1] - values[0]
		val = _adjust_modulo(val, modulo=span, min=mi)
		assert val >= mi and val <= ma, "pb wrap"

	    else:
		raise ValueError("invalid parameter: mode="+repr(mode))

    loc = np.argmin(np.abs(val-values))
    if np.abs(values[loc]-val) > tol:
	raise ValueError("%f not found within tol %f (closest match %i:%f)" % (val, tol, loc, values[loc]))

    return loc


def _adjust_modulo(val, modulo, min=0):
    oldval = val
    mval = np.mod(val, modulo)
    mmin = np.mod(min, modulo)
    if mval < mmin:
	mval += modulo
    val = min + (mval - mmi)
    assert np.mod(val-oldval, modulo) == 0, "pb modulo"
    return val


#
# Return a slice for an axis
#
class Locator_axis(Config):
    """ This class is the core of indexing in dimarray. 

	loc = Locator_axis(values, **opt)  

    where `values` represent the axis values


    A locator instance is generated from within the Axis object, via 
    its properties loc (valued-based indexing) and iloc (integer-based)

	axis.loc  ==> Locator_axis(values)  

    A locator is hashable is a similar way to a numpy array, but also 
    callable to update parameters on-the-fly.

    It returns an integer index or `list` of `int` or `slice` of `int` which 
    is understood by numpy's arrays. In particular we have:

	loc[ix] == np.index_exp[loc[ix]][0]

    The "set" method can also be useful for chained calls. We have the general 
    equivalence:

	loc(idx, **kwargs) :: loc.set(**kwargs)[idx]

    Examples:
    ---------
    >>> values = np.arange(1950.,2000.)
    >>> values  # doctest: +ELLIPSIS
    array([ 1950., ... 1999.])
    >>> loc = Locator_axis(values)   
    >>> loc(1951) 
    1
    >>> loc(1951.4, mode="nearest")   # also works in nearest approx
    1
    >>> loc([1960, 1980, 1999])		# a list if also fine 
    [10, 30, 49]
    >>> loc((1960,1970))		# or a tuple/slice (latest index included)
    slice(10, 21, None)
    >>> loc[1960:1970] == _		# identical, as any of the commands above
    True

    Test equivalence with np.index_exp
    >>> ix = 1951
    >>> loc[ix] == np.index_exp[loc[ix]][0]
    True
    >>> ix = [1960, 1980, 1999]
    >>> loc[ix] == np.index_exp[loc[ix]][0]
    True
    >>> ix = slice(1960,1970)
    >>> loc[ix] == np.index_exp[loc[ix]][0]
    True
    >>> ix = 1951
    >>> loc[ix] == np.index_exp[loc[ix]][0]
    True
    """
    mode = "exact"   # "numpy", "exact", "nearest", "interp"
    tol = 1e-8	     # tolerance to find float
    bounds = "raise" # out-of-bound errors
    repna = False # replace by NaN instead of raising an Error ?
    nan = np.nan # replacement value if not raise_error
    modulo = None # cyclic axis with period `modulo`
    keepdims = False # if True, always return an array (and no scalar)
    def __init__(self, values, **opt):
	"""
	values	: string list or numpy array
	nan : replacement value in case of ValueError

	see help on locate_num
	"""
	self.values = values
	for o in opt: if o not in self: raise ValueError("unknown option {}".format(o))
	self.__dict__.update(opt) # update default options

    #
    # wrapper mode: __getitem__ and __call__
    #
    def __getitem__(self, ix):
	""" 
	"""
	#
	# check special cases
	#
	# do not do anything in numpy mode
	if self.numpy:
	    return ix

	# ...nor with boolean indexing
	if type(ix) is np.ndarray and ix.dtype is np.dtype(bool):
	    return ix

	# make sure (1,) is understood as 1 just as numpy would
	if type(ix) is tuple:
	    if len(ix) == 1:
		ix = ix[0]
	    else:
		raise ValueError("dimension mismatch")

	#
	# look up corresponding numpy indices
	#
	# e.g. 45:56
	if type(ix) is slice:
	    return self.slice(ix)

	# int, float, string
	elif np.isscalar(ix):
	    return self.locate(ix)

	# list, array
	elif np.iterable(ix):
	    return self.take(ix)

	else:
	    raise TypeError("unknown type: "+repr(ix))

    def __call__(self, idx, **kwargs):
	""" general wrapper method
	
	input:
	    idx: int, list, slice, tuple (on integer index or axis values)
	    **kwargs: see help on Locator_axis

	return:
	    `int`, list of `int` or slice of `int`
	
	"""
	#if method is None: method = self.method
	loc = self.set(**kwargs)
	if self.keepdims and np.isscalar(idx):
	    idx = [idx]
	return loc[idx]

    def set(self, **kwargs):
	""" convenience function for chained call: update methods and return itself 
	"""
	#self.method = method
	return Locator_axis(self.values, **kwargs)

    #
    # wrapper for single value
    #
    def locate(self, val, mode=None):
	""" locate with try/except checks
	"""
	if mode is None: mode = self.mode

	try:
	    # interp mode, return fractional index
	    if mode == "interp":
		res = self.interp(val)
	    else:
		res = self._locate(val, mode)

	except ValueError, msg:
	    if self.repna:
		res = self.nan
	    else:
		raise

	return res

    def _locate(self, val, mode):
	""" locate without try/except check
	"""
	# nearest mode, increase tolerance
	if mode == "nearest":
	    tol = np.inf
	else:
	    tol = self.tol

	# use list's index for object types
	if self.values.dtype is np.dtype('O'):
	    return self.values.tolist().index(val)
	else:
	    return locate_num(self.values, val, mode=self.bounds, tol=tol, modulo=self.modulo)

    #
    # Access a list or array
    #
    def locate_list(self, indices, **kwargs):
	""" Return a list of indices

	**kwargs: passed to locate
	"""
	#assert type(indices) is list, "must provide a list !"
	assert np.iterable(indices), "indices is not iterable!"
	locate = functools.partial(self.locate, **kwargs)
	return map(locate, indices)

    #
    # Access a slice
    #
    def slice(self, slice_, mode=None, include_last=True):
	""" Return a slice_ object

	slice_	    : slice or tuple 
	include_last: include last element 

	Note bound checking is automatically done via "locate" mode
	This is in contrast with slicing in numpy arrays.
	"""
	# Check type
	if type(slice_) is not slice:
	    raise TypeError("should be slice !")

	# update mode
	if mode is None: mode = self.mode

	start, stop, step = slice_.start, slice_.stop, slice_.step

	if start is not None:
	    start = self.locate(start, mode=mode)

	if stop is not None:
	    stop = self.locate(stop, mode=mode)
	    
	    #at this stage stop is an integer index on the axis, 
	    # so make sure it is included in the slice if required
	    if include_last:
		stop += 1

	# leave the step unchanged: it always means subsampling
	return slice(start, stop, step)

    #
    # interp mode: return a fractional index
    #
    def interp(self, val):
	""" return fractional index for interpolation
	"""
	assert is_regular(self.values), "interp mode only makes sense for regular axes !"
	# index of nearest neighbour
	if np.iterable(val):
	    i = self.locate_list(val, mode="nearest")
	else:
	    i = self.locate(val, mode="nearest")
	# and corresponding value
	xi = self.values[i]
	# axis step
	dx = float(self.values[1] - self.values[0])
	return i + (val-xi)/dx


    @property
    def size(self):
	return np.size(self.values)


class Locator_axes(object):
    """ return indices over multiple axes
    """
    def __init__(self, axes, **opt):
	"""
	"""
	assert type(axes) is list and isinstance(axes[0], Axis), "must be list of axes objects"
	self.axes = axes
	self.opt = opt

    def set(self, **kwargs):
	""" convenience function for chained call: update methods and return itself 
	"""
	return Locator_axes(self.axes, **kwargs)

    def __getitem__(self, indices):
	"""
	"""
	# Construct the indices
	indices = np.index_exp[indices]
	numpy_indices = ()
	for i, ix in enumerate(indices):
	    numpy_indices += Locator_axis(self.axes[i].values, **self.opt)[ix]

	# Just add slice(None) if some indices are missing
	for i in range(len(axes) - len(numpy_indices)):
	    numpy_indices += slice(None)

	return numpy_indices

    def __call__(self, indices=None, axis=0, **opt):
	"""
	"""
	self = self.set(**opt)

	# dict: just convert to appropriately ordered tuple
	if isinstance(indices, dict):
	    kw = indices
	    indices = ()
	    for ax in self.axes:
		if ax.name in kw:
		    ix = kw[ax.name]
		else:
		    ix = slice(None)
		indices += ix,

	if type(indices) is tuple:
	    assert axis in (None, 0), "cannot have axis > 0 for tuple (multi-dimensional) indexing"
	    return self[indices]

	# along-axis slicing
	return Locator_axis(self.axes[axis].values, **self.opt)[indices]


def take(self, ix=None, axis=0, mode=None, bounds="raise", repna=True, tol=1e-8, keepdims=False):
    """ Retrieve values from a DimArray

    input:

	- ix       : int or list or slice (single-dimensional indices)
	             or a tuple of those (multi-dimensional)
		     or `dict` (`axis name` : `indices`)
	- axis     : int or str
	- mode     : "numpy", "exact", "nearest", "interp" 
	- bounds   : "raise", "clip", "wrap"
		     "clip" and "wrap" only valid for regular axes
		     analogous to numpy, see help on np.take
	- repna    : replace missing values with NaNs?
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
    >>> v.take({'x0':'a', 'x1':10})  # dict-like arguments
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

    Increase tolerance for floats
    >>> v.take(12, axis="d1", tol=5)
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0'
    0 / d0 (2): a to b
    array([ 1.,  4.])
    """
#
#    Replace missing values with NaNs
#    >>> v.take(([12,20], axis="d1", mode="flag")
#    dimarray: 2 non-null elements (0 null)
#    dimensions: 'd0'
#    0 / d0 (2): a to b
#    array([[nan,  2.],
#	   [nan,  5.]])
    if mode is None:
	mode = self._indexing

    return _take_check(self, ix=ix, axis=axis, mode=mode, keepdims=keepdims, raise_error=raise_error, flag=flag)

def put(obj, val, ix, axis=0, mode=None):
    """ Put new values into DimArray (inplace)

    parameters:
    -----------
    obj: DimArray (do not provide if method bound to class instance)
    val: value to put in: scalar or array-like with appropriate shape
    ix: single- or multi- or `bool` index
    axis: for single index (see help on `take` method and axes Locator)
    mode: see `take`

    returns:
    --------
    None: (inplace modification)

    Examples:
    ---------
    """
    if mode is None:
	mode = obj._indexing

    # locate indices
    multi_index = Locator_axes(obj.axes, mode=mode)(ix, axis=axis)
    obj.values[multi_index] = val 

def _take(obj, indices, **kwargs):
    """ take indices from one of the standard modes ("numpy", "exact", "nearest")
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


def _take_check(obj, indices, mode="exact", **kwargs):
    """ same as take but also nans indices
    """
    # Convert to a tuple numpy indices matching the shape of the DimArray
    indices_numpy = Locator_axes(obj.axes, **kwargs)(indices)

    # Filter bad from good indices: this would happen only for raise_error == False
    indices_numpy, bad_ix = _get_bad_indices(indices_numpy)

    # Pick-up values 
    if mode == "interp":
	result = _take_interp(obj, indices_numpy)
	#    # fix type
	#    ix = np.asarray(ix) # make sure ix is an array
	#    if not np.isscalar(result):
	#	result.axes[axis].values = np.array(result.axes[axis].values, dtype=ix.dtype)

    else:
	result = _take(obj, indices_numpy, mode="numpy")

    if bad_ix is not None:
	result = _fill_bad_indices(result, bad_ix, indices)

    return result

def _get_bad_indices(indices_numpy, mode):
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
	    if not mode == "interp":
		ind = np.array(ind, dtype=int) # NaNs produced conversion to float
	    bads[i] = bad # record bad indices
	indices_nanfree[i] = ind

    return indices_nanfree, bads

def _fill_bad_indices(result, bad_ix, indices, flag=np.nan):
    """ fill NaN back in

    input: 
	result: DimArray
	bad_ix: `dict` of `bool` 1-D arrays to indicate the locations of bad indices
	indices: originally required indices (n-d)
	flag: replacement values for bad indices (default NaNs)

    output:
	result: corrected DimArray with bad values replaced with `flag` 
    """
    for k in bad_ix:
	bad = bad_ix[k] # `bool` indices of bad numbers

	# replace with originally asked-for values
	result.axes[k].values[bad] = indices[bad] 

	# replace array values with NaNs 
	pos = result.dims.index(k) # index of the new axis position
	ix = pos*(slice(None),) + bad, # multi-index of bad positions

	# convert to float 
	if result.dtype is not np.dtype(flag):
	    result.values = np.asarray(result.values, dtype=np.dtype(flag))
	result.values[ix] = flag
	#result.put(flag, bad_ix[k], axis=k)



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

def reindex_axis(self, values, axis=0, method='exact', repna=True):
    """ reindex an array along an axis

    Input:
	- values : array-like or Axis: new axis values
	- axis   : axis number or name
	- method : "exact" (default), "nearest", "interp" (see take)
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
    if method in ("nearest","exact","interp", None):
	newobj = self.take(values, axis=axis, mode=method, repna=repna)

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


#
# 1-D interpolation
#
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
    v0 = take(obj, i0, axis=axis, mode="numpy")
    v1 = take(obj, i1, axis=axis, mode="numpy")

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

