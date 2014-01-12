""" array with physical dimensions (named and valued axes)
"""
import numpy as np
import copy
import warnings
from collections import OrderedDict as odict

from dimarray.config import Config

from metadata import Metadata
from axes import Axis, Axes, GroupedAxis

import _transform  # numpy along-axis transformations, interpolation
import _reshape	   # change array shape and dimensions
import _indexing   # perform slicing and indexing operations
import _operation  # operation between DimArrays
import missingvalues # operation between DimArrays

from tools import pandas_obj, is_array1d_equiv

__all__ = ["DimArray", "array"]

class DimArray(Metadata):
    """ numpy's ndarray with physical dimensions (named and values axes)

    Parameters:
    -----------

    values	: numpy-like array, or DimArray instance
		  If `values` is not provided, will initialize an empty array 
		  with dimensions inferred from axes (in that case `axes=` 
		  must be provided).

    axes	: optional, list or tuple: axis values as ndarrays, whose order 
		  matches axis names (the dimensions) provided via `dims=` 
		  parameter. Each axis can also be provided as a tuple 
		  (str, array-like) which contains both axis name and axis 
		  values, in which case `dims=` becomes superfluous.
		  `axes=` can also be provided with a list of Axis objects
		  If `axes=` is omitted, a standard axis `np.arange(shape[i])`
		  is created for each axis `i`.

    dims	: optional, list or tuple: dimensions (or axis names)
		  This parameter can be omitted if dimensions are already 
		  provided by other means, such as passing a list of tuple 
		  to `axes=`. If axes are passed as keyword arguments (via 
		  **kwargs), `dims=` is used to determine the order of 
		  dimensions. If `dims` is not provided by any of the means 
		  mentioned above, default dimension names are 
		  given `x0`, `x1`, ...`xn`, where n is the number of 
		  dimensions.

    dtype	: optional, data type, passed to np.array() 
    copy	: optional, passed to np.array()

    **kwargs	: metadata

		  NOTE: metadata passed this way cannot have name already taken by other 
		  parameters such as "values", "axes", "dims", "dtype" or "copy".
		  See setncattr for such special cases.

    Examples:
    ---------

    Basic:

    >>> DimArray([[1,2,3],[4,5,6]]) # automatic labelling
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[1, 2, 3],
	   [4, 5, 6]])

    >>> DimArray([[1,2,3],[4,5,6]], dims=['items','time'])  # axis names only
    dimarray: 6 non-null elements (0 null)
    dimensions: 'items', 'time'
    0 / items (2): 0 to 1
    1 / time (3): 0 to 2
    array([[1, 2, 3],
	   [4, 5, 6]])

    >>> DimArray([[1,2,3],[4,5,6]], axes=[list("ab"), np.arange(1950,1953)]) # axis values only
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): a to b
    1 / x1 (3): 1950 to 1952
    array([[1, 2, 3],
	   [4, 5, 6]])

    More general case:

    >>> a = DimArray([[1,2,3],[4,5,6]], axes=[list("ab"), np.arange(1950,1953)], dims=['items','time']) 
    >>> b = DimArray([[1,2,3],[4,5,6]], axes=[('items',list("ab")), ('time',np.arange(1950,1953))])
    >>> c = DimArray([[1,2,3],[4,5,6]], {'items':list("ab"), 'time':np.arange(1950,1953)}) # here dims can be omitted because shape = (2, 3)
    >>> np.all(a == b == c)
    True
    >>> a
    dimarray: 6 non-null elements (0 null)
    dimensions: 'items', 'time'
    0 / items (2): a to b
    1 / time (3): 1950 to 1952
    array([[1, 2, 3],
	   [4, 5, 6]])

    Empty data

    >>> a = DimArray(axes=[('items',list("ab")), ('time',np.arange(1950,1953))])

    Metadata

    >>> a = DimArray([[1,2,3],[4,5,6]], name='test', units='none') 
    
    Operations

    >>> a = da.DimArray([[1,2,3],[3,4,5]],dims=('x0','x1'))
    >>> a == a
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> np.all(a == a)
    True
    >>> np.all(a+2 == a + np.ones(a.shape)*2)
    True
    >>> np.all(a+a == a*2)
    True
    >>> np.all(a*a == a**2)
    True
    >>> np.all((a - a.values) == a - a)
    True

    Broadcasting (dimension alignment)
    >>> x = da.DimArray(np.arange(2), dims=('x0',))
    >>> y = da.DimArray(np.arange(3), dims=('x1',))
    >>> x+y
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[0, 1, 2],
           [1, 2, 3]])
    
    Reindexing (axis values alignment)
    >>> z = da.DimArray([0,1,2],('x0',[0,1,2]))
    >>> x+z
    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  0.,   2.,  nan])
    
    Broadcasting + Reindexing
    >>> (x+y)+(x+z)
    dimarray: 6 non-null elements (3 null)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (3): 0 to 2
    array([[  0.,   1.,   2.],
           [  3.,   4.,   5.],
           [ nan,  nan,  nan]])
    """
    _order = None  # set a general ordering relationship for dimensions
    _metadata_exclude = ("values","axes") # is NOT a metadata

    #
    # NOW MAIN BODY OF THE CLASS
    #

    def __init__(self, values=None, axes=None, dims=None, copy=False, dtype=None, _indexing="values", **kwargs):
	""" Initialization. See help on DimArray.
	"""
	# check if attached to values (e.g. DimArray object)
	if hasattr(values, "axes") and axes is None:
	    axes = values.axes

	#
	# array values
	#
	if values is not None:
	    values = np.array(values, copy=copy, dtype=dtype)

	# special case: 1D object: accept single axis instead of list of axes/dimensions
	if values is not None and values.ndim == 1:
	    
	    # accept a tuple ('dim', axis values) instead of [(...)]
	    if type(axes) is tuple:
		if len(axes) == 2 and type(axes[0]) is str and is_array1d_equiv(axes[1]):
		    axes = [axes]

	    # accept axes=axis, dims='dim' (instead of list)
	    elif (axes is None or is_array1d_equiv(axes)) and (type(dims) in (str, type(None))):
		#if axes is not None: assert np.size(axes) == values.size, "invalid argument: bad size"
		axes = [axes] if axes is not None else None
		dims = [dims] if dims is not None else None

	#
	# Initialize the axes
	# 
	# Can be one of
	# - dict
	# - list of Axis objects
	# - list of tuples `dim, array`
	# - list of arrays, to be complemented by "dims="
	# - nothing

	# axis not provided: check whether values has an axes field
	if axes is None:
	    assert values is not None, "values= or/and axes= required"

	    # check if attached to values (e.g. DimArray or pandas object)
	    if hasattr(values, "axes"):
		axes = values.axes

	    # define a default set of axes if not provided
	    axes = Axes.from_shape(values.shape, dims=dims)

	elif isinstance(axes, dict):
	    kwaxes = axes
	    if values is None: shape = None
	    else: shape = values.shape
	    if isinstance(kwaxes, odict):
		dims = kwaxes.keys()
	    axes = Axes.from_kw(dims=dims, shape=shape, kwaxes=kwaxes)
	    assert type(axes) is Axes

	# list of Axis objects
	elif isinstance(axes[0], Axis):
	    axes = Axes(axes)

	# (name, values) tuples
	elif isinstance(axes[0], tuple):
	    axes = Axes.from_tuples(*axes)

	# axes contains only axis values, with names possibly provided in `dims=`
	#elif is_array_equiv(axes[0]): 
	elif type(axes[0]) in (list, np.ndarray): 
	    axes = Axes.from_arrays(axes, dims=dims)

	else:
	    raise TypeError("axes, if provided, must be a list of: `Axis` or `tuple` or arrays. Got: {}".format(axes))

	assert type(axes) is Axes

	# if values not provided, create empty data (filled with NaNs if dtype is float, -999999 for int)
	if values is None:
	    values = np.empty([ax.size for ax in axes], dtype=dtype)
	    if dtype in (float, None, np.dtype(float)):
		values.fill(np.nan)
	    else:
		warnings.warn("no nan representation for {}, array left empty".format(repr(dtype)))

	#
	# store all fields
	#

	# init an ordered dict self._metadata
	Metadata.__init__(self)
	#super(DimArray, self).__init__() 

	self.values = values
	self.axes = axes

	## options
	self._indexing = _indexing

	#
	# metadata (see Metadata type in metadata.py)
	#
	for k in kwargs:
	    self.setncattr(k, kwargs[k]) # perform type-checking and store in self._metadata

	# Check consistency between axes and values
	inferred = tuple([ax.size for ax in self.axes])
	if inferred != self.values.shape:
	    msg = """\
shape inferred from axes: {}
shape inferred from data: {}
mismatch between values and axes""".format(inferred, self.values.shape)
	    raise Exception(msg)

	# If a general ordering relationship of the class is assumed,
	# always sort the class
	if self._order is not None and self.dims != tuple(dim for dim in self._order if dim in self.dims):
	    present = filter(lambda x: x in self.dims, self._order)  # prescribed
	    missing = filter(lambda x: x not in self._order, self.dims)  # not
	    order = missing + present # prepend dimensions not found in ordering relationship
	    obj = self.transpose(order)
	    self.values = obj.values
	    self.axes = obj.axes

    @classmethod
    def from_kw(cls, *args, **kwargs):
	""" Alternative definition of a Dimarray which allow keyword arguments 
	in addition to other methods, but at the expense of metadata and other parameters

	*args : [values, [dims,]]
	**kwargs	: axes as keyword arguments

	Notes:
	------

	The key-word functionality comes at the expense of metadata, which needs to be 
	added after creation of the DimArray object.

        If axes as passed as kwargs, also needs to be provided
        or an error will be raised, unless values's shape is 
        sufficient to determine ordering (when all axes have different 
        sizes).  This is a consequence of the fact 
        that keyword arguments are *not* ordered in python (any order
        is lost since kwargs is a dict object)

        Axes passed by keyword arguments cannot have name already taken by other 
        parameters such as "values", "axes", "dims", "dtype" or "copy"

	Examples: 
	---------
	(da.array is an alias for DimArray.from_kw)
	>>> a = da.DimArray.from_kw([[1,2,3],[4,5,6]], items=list("ab"), time=np.arange(1950,1953)) # here dims can be omitted because shape = (2, 3)
	>>> b = da.DimArray.from_kw([[1,2,3],[4,5,6]], ['items','time'], items=list("ab"), time=np.arange(1950,1953)) # here dims can be omitted because shape = (2, 3)
	>>> c = da.DimArray([[1,2,3],[4,5,6]], {'items':list("ab"), 'time':np.arange(1950,1953)}) # here dims can be omitted because shape = (2, 3)
	>>> np.all(a == b == c)
	True

	See also DimArray's doc for more examples
	"""
	if len(args) == 0:
	    values = None
	else:
	    values = args[0]

	if len(args) > 1:
	    dims = args[1]
	else:
	    dims = None

	axes = {k:kwargs[k] for k in kwargs}

	assert len(args) <= 2, "[values, [dims]]"

	return cls(values, axes, dims)

    #
    # Internal constructor, useful for subclassing
    #
    @classmethod
    def _constructor(cls, values, axes, **metadata):
	""" Internal API for the constructor: check whether a pre-defined class exists

	values	: array-like
	axes	: Axes instance 

	This static method is used whenever a new DimArray needs to be instantiated
	for example after a transformation.

	This makes the sub-classing process easier since only this method needs to be 
	overloaded to make the sub-class a "closed" class.
	"""
	#if '_indexing' in metadata:
	#    kwargs['_indexing'] = metadata.pop('_indexing')
	#obj = cls(values, axes)
	#for k in metadata: obj.setncattr(k, metadata[k])
	return cls(values, axes, **metadata)

    #
    # Attributes access
    #
    def __getattr__(self, att):
	""" return dimension or numpy attribue
	"""
	# check for dimensions
	if att in self.dims:
	    ax = self.axes[att]
	    return ax.values # return numpy array

	else:
	    try:
		return super(DimArray, self).__getattr__(att) # call Metadata's method

	    except AttributeError, msg:
		raise AttributeError(msg)

	    else:
		raise

    def copy(self, shallow=False):
	""" copy of the object and update arguments

	shallow: if True, does not copy values and axes
	"""
	import copy
	new = copy.copy(self) # shallow copy

	if not shallow:
	    new.values = self.values.copy()
	    new.axes = self.axes.copy()

	return new
	#return DimArray(self.values.copy(), self.axes.copy(), slicing=self.slicing, **{k:getattr(self,k) for k in self.ncattrs()})

    #
    # Basic numpy array's properties
    #
    # basic numpy shape attributes as properties
    @property
    def shape(self): 
	return self.values.shape

    @property
    def size(self): 
	return self.values.size

    @property
    def ndim(self): 
	return self.values.ndim

    @property
    def dtype(self): 
	return self.values.dtype

    @property
    def __array__(self): 
	return self.values.__array__

    @property
    def __iter__(self): 
	""" iterates on values, consistently with a ndarray
	"""
	for k, val in self.iter():
	    yield val

    #
    # Some additional properties
    #
    @property
    def dims(self):
	""" axis names 
	"""
	return tuple([ax.name for ax in self.axes])

    #
    # iteration over any dimension: key, value
    #
    def iter(self, axis=0):
	""" Iterate over axis value and slice along an axis

	for time, time_slice in myarray.iter('time'):
	    do stuff
	"""
	# iterate over axis values
	for k in self.axes[axis].values:
	    val = self.take(k, axis=axis) # cross-section
	    yield k, val

    #
    # returns axis position and name based on either of them
    #
    def _get_axis_info(self, axis):
	""" axis position and name

	input  : 
	    axis: `int` or `str` or None

	returns: 
	    idx	: `int`, axis position
	    name: `str` or None, axis name
	"""
	if axis is None:
	    return None, None

	if type(axis) is str:
	    idx = self.dims.index(axis)

	elif type(axis) is int:
	    idx = axis

	else:
	    raise TypeError("axis must be int or str, got:"+repr(axis))

	name = self.axes[idx].name
	return idx, name

    #
    # Return a weight for each array element, used for `mean`, `std` and `var`
    #
    def get_weights(self, weights=None, axis=None, fill_nans=True):
	""" Weights associated to each array element
	
	optional arguments: 
	    weights: (numpy)array-like of weights, taken from axes if None or "axis" or "axes"
	    axis   : int or str, if None a N-D weight is created

	return:
	    weights: DimArray of weights, or None

	NOTE: this function is used by `mean`, `std` and `var`
	"""
	# make sure weights is a nd-array
	if weights is not None:
	    weights = np.array(weights) 

	if weights in ('axis','axes'):
	    weights = None

	if axis is None:
	    dims = self.dims

	elif type(axis) is tuple:
	    dims = axis

	else:
	    dims = (axis,)

	# axes over which the weight is defined
	axes = [ax for ax in self.axes if ax.name in dims]

	# Create weights from the axes if not provided
	if weights is None:

	    weights = 1
	    for axis in dims:
		ax = self.axes[axis]
		if ax.weights is None: continue
		weights = ax.get_weights().reshape(dims) * weights

	    if weights is 1:
		return None
	    else:
		weights = weights.values

	# Now create a dimarray and make it full size
	# ...already full-size
	if weights.shape == self.shape:
	    weights = DimArray(weights, self.axes)

	# ...need to be expanded
	elif weights.shape == tuple(ax.size for ax in axes):
	    weights = DimArray(weights, axes).expand(self.dims)

	else:
	    try:
		weights = weights + np.zeros_like(self.values)

	    except:
		raise ValueError("weight dimensions are not conform")
	    weights = DimArray(weights, self.axes)

	#else:
	#    raise ValueError("weight dimensions not conform")

	# fill NaNs in when necessary
	if fill_nans:
	    weights.values[np.isnan(self.values)] = np.nan
	
	assert weights is not None
	return weights


    #
    # INDEXING
    #

    #
    # New general-purpose indexing method
    #
    take = _indexing.take
    put = _indexing.put  

    # get the kw index equivalent {dimname:indices}
    _get_keyword_indices = _indexing._get_keyword_indices 

    #
    # Standard indexing takes axis values
    #
    def __getitem__(self, indices): 
	""" get a slice (use take method)
	"""
	#indexing = self.axes.loc[indices]
	#return self.take(indexing, position_index=True)
	return self.take(indices, indexing=self._indexing)

    def __setitem__(self, ix, val):
	"""
	"""
	self.put(val, ix, indexing=self._indexing, inplace=True)

    # 
    # Can also use integer-indexing via ix
    #
    @property
    def ix(self):
	""" integer index-access (toogle between integer-based and values-based indexing)
	"""
	if self._indexing is "values": 
	    _indexing = "position"
	else:
	    _indexing = "values"
	return self._constructor(self.values, self.axes, _indexing=_indexing , **self._metadata)

    #
    # TRANSFORMS
    # 

    #
    # ELEMENT-WISE TRANSFORMS
    #
    def apply(self, func, args=(), **kwargs):
	""" Apply element-wise function to DimArray
	"""
	return self._constructor(func(self.values, *args, **kwargs), self.axes.copy())

    #
    # NUMPY TRANSFORMS
    #
    median = _transform.median
    sum = _transform.sum
    prod = _transform.prod

    # use weighted mean/std/var by default
    mean = _transform.weighted_mean
    std = _transform.weighted_std
    var = _transform.weighted_var

    #_mean = _transform.mean
    #_var = _transform.var
    #_std = _transform.std

    all = _transform.all
    any = _transform.any

    min = _transform.min
    max = _transform.max
    ptp = _transform.ptp

    #
    # change `arg` to `loc`, suggesting change in indexing behaviour
    #
    locmin = _transform.locmin
    locmax = _transform.locmax

    cumsum = _transform.cumsum
    cumprod = _transform.cumprod
    diff = _transform.diff

    #
    # METHODS TO CHANGE ARRAY SHAPE AND SIZE
    #
    #from _reshape import repeat, newaxis, transpose, reshape, broadcast, group, ungroup, squeeze
    repeat = _reshape.repeat
    newaxis = _reshape.newaxis
    squeeze = _reshape.squeeze
    transpose = _reshape.transpose
    reshape = _reshape.reshape
    broadcast = _reshape.broadcast
    group = _reshape.group
    ungroup = _reshape.ungroup
    groupby = _reshape.groupby
    
    @property
    def T(self):
	return self.transpose()


    #
    # REINDEXING 
    #
    reindex_axis = _indexing.reindex_axis
    reindex_like = _indexing.reindex_like

    # Drop missing values
    dropna = missingvalues.dropna

    #
    # BASIC OPERATTIONS
    #
    def _operation(self, func, other):
	""" make an operation: this include axis and dimensions alignment

	Just for testing:
	>>> b = DimArray([[0.,1],[1,2]])
	>>> b
	... # doctest: +SKIP
	array([[ 0.,  1.],
	       [ 1.,  2.]])
	>>> np.all(b == b)
	True
	>>> np.all(b+2 == b + np.ones(b.shape)*2)
	True
	>>> np.all(b+b == b*2)
	True
	>>> np.all(b*b == b**2)
	True
	>>> np.all((b - b.values) == b - b)
	True
	"""
	result = _operation.operation(func, self, other, broadcast=Config.op_broadcast, reindex=Config.op_reindex, constructor=self._constructor)
	return result

    def __add__(self, other): return self._operation(np.add, other)

    def __sub__(self, other): return self._operation(np.subtract, other)

    def __mul__(self, other): return self._operation(np.multiply, other)

    def __div__(self, other): return self._operation(np.divide, other)

    def __pow__(self, other): return self._operation(np.power, other)
    def __sqrt__(self, other): return self**0.5

    def __float__(self):  return float(self.values)
    def __int__(self):  return int(self.values)

    def _to_array_equiv(self, other):
	""" convert to equivalent numpy array
	raise Exception is other is a DimArray with different axes
	"""
	if isinstance(other, DimArray):
	    # check if axes are equal
	    if self.axes != other.axes:
		raise ValueError("axes differ !")

	return np.asarray(other)

    def __eq__(self, other): 
	""" equality in numpy sense

	>>> test = DimArray([1, 2]) == 1
	>>> test
	dimarray: 2 non-null elements (0 null)
	dimensions: 'x0'
	0 / x0 (2): 0 to 1
	array([ True, False], dtype=bool)
	>>> test2 = DimArray([1, 2]) == DimArray([1, 2])
	>>> test3 = DimArray([1, 2]) == np.array([1, 2])
	>>> np.all(test == test2 == test3)
	True
	>>> DimArray([1, 2]) == DimArray([1, 2],dims=('x1',))
	False
	"""
	# check axes and convert to arrays
	try: 
	    other = self._to_array_equiv(other)
	except ValueError:
	    return False

	res = self.values == other

	if isinstance(res, np.ndarray):
	    return self._constructor(res, self.axes)

	# boolean False
	else:
	    return res

    def __invert__(self): 
	"""
	Examples:
	>>> a = DimArray([True, False])
	>>> ~a
	dimarray: 2 non-null elements (0 null)
	dimensions: 'x0'
	0 / x0 (2): 0 to 1
	array([False,  True], dtype=bool)
	"""
	return self._constructor(np.invert(self.values), self.axes)

    def _cmp(self, op, other):
	""" Element-wise comparison operator

	>>> a = DimArray([1, 2])
	>>> a < 2
	dimarray: 2 non-null elements (0 null)
	dimensions: 'x0'
	0 / x0 (2): 0 to 1
	array([ True, False], dtype=bool)

	>>> b = DimArray([1, 2], dims=('x1',))
	>>> try: 
	...	a > b 
	... except ValueError, msg:
	...	print msg	
	axes differ !
	"""
	other = self._to_array_equiv(other)
	fop = getattr(self.values, op)
	return self._constructor(fop(other), self.axes)

    def __lt__(self, other): return self._cmp('__lt__', other)
    def __le__(self, other): return self._cmp('__le__', other)
    def __gt__(self, other): return self._cmp('__gt__', other)
    def __ge__(self, other): return self._cmp('__ge__', other)

    @classmethod
    def from_dict(cls, data, keys=None, axis="items"):
	""" initialize a DimArray from a dictionary of smaller dimensional DimArray

	Convenience method for: collect.Dataset(data, keys).to_array(axis)

	input:
	    - data : list or dict of DimArrays
	    - keys, optional : ordering of data (for dict)
	    - dim, optional : dimension name along which to aggregate data (default "items")

	output:
	    - new DimArray object, with axis alignment (reindexing)

	Examples:
	---------

	>>> d = {'a':DimArray([10,20,30.],[0,1,2]), 'b':DimArray([1,2,3.],[1.,2.,3])}
	>>> a = DimArray.from_dict(d, keys=['a','b']) # keys= just needed to enforce ordering
	>>> a
	dimarray: 6 non-null elements (2 null)
	dimensions: 'items', 'x0'
	0 / items (2): a to b
	1 / x0 (4): 0 to 3
	array([[ 10.,  20.,  30.,  nan],
	       [ nan,   1.,   2.,   3.]])
	"""
	from dimarray.dataset import Dataset
	data = Dataset(data, keys=keys)
	return data.to_array(axis=axis)

    #
    # export to other data types
    #

    # Split along an axis
    def to_odict(self, axis=0):
	"""
	"""
	d = odict()
	for k, val in self.iter(axis):
	    d[k] = val
	return d

    def to_dataset(self, axis=0):
	""" split a DimArray into a Dataset object (collection of DimArrays)
	"""
	from dimarray.dataset import Dataset
	# iterate over elements of one axis
	data = [val for k, val in self.iter(axis)]
	return Dataset(data)

    def to_MaskedArray(self):
	""" transform to MaskedArray, with NaNs as missing values
	"""
	return _transform.to_MaskedArray(self.values)

    def to_list(self, axis=0):
	return [val for  k, val in self.iter(axis)]

    def to_pandas(self):
	""" return the equivalent pandas object
	"""
	obj = pandas_obj(self.values, *[ax.values for ax in self.axes])

	# make sure the axes have the right name
	for i, ax in enumerate(self.axes):
	    obj.axes[i].name = ax.name

	return obj

    def to_larry(self):
	""" return the equivalent pandas object
	"""
	import la
	a = la.larry(self.values, [list(ax.values) for ax in self.axes])
	print "warning: dimension names have not been passed to larry"
	return a

    #
    # pretty printing
    #

    def __repr__(self):
	""" pretty printing
	"""
	if self.ndim > 0:
	    nonnull = np.size(self.values[~np.isnan(self.values)])
	else:
	    nonnull = ~np.isnan(self.values)

	lines = []

	#if self.size < 10:
	#    line = "dimarray: "+repr(self.values)
	#else:
	line = "dimarray: {} non-null elements ({} null)".format(nonnull, self.size-nonnull)
	lines.append(line)

	# # show metadata as well?
	# If len(self.ncattrs()) > 0:
	#     line = self.repr_meta()
	#     lines.append(line)

	if True: #self.size > 1:
	    line = repr(self.axes)
	    lines.append(line)

	if self.size < Config.display_max:
	    line = repr(self.values)
	else:
	    line = "array(...)"
	lines.append(line)

	return "\n".join(lines)


    #
    #  I/O
    # 

    def write_nc(self, f, name=None, *args, **kwargs):
	""" Write to netCDF
	"""
	import dimarray.io.nc as ncio

	# add variable name if provided...
	if name is None and hasattr(self, "name"):
	    name = self.name

	ncio.write_variable(f, self, name, *args, **kwargs)

    @classmethod
    def read_nc(cls, f, *args, **kwargs):
	""" Read from netCDF
	"""
	import dimarray.io.nc as ncio
	return ncio.read_base(f, *args, **kwargs)

    # Aliases
    write = write_nc
    read = read_nc

    #
    # Plotting
#    #
#    def scatter_vs(self, axis=0, **kwargs):
#	""" make a plot along the first dim
#	"""
#	if len(self.dims) > 1:
#	    raise NotImplementedError("plot only valid grouped by 1 variable")
#	import matplotlib.pyplot as plt
#	if 'ax' in kwargs:
#	    ax = kwargs.pop('ax')
#	else:
#	    ax = plt.gca()
#
#	return ax.plot(self.a.axes[0].values, *args, **kwargs)

#    def plot(self, *args, **kwargs):
#	""" by default, use pandas for plotting
#	"""
#	assert self.ndim <= 2, "only support plotting for 1- and 2-D objects"
#	return self.to_pandas().plot(*args, **kwargs)

def array(*args, **kwargs):
    return DimArray.from_kw(*args, **kwargs)

array.__doc__ = DimArray.from_kw.__doc__
