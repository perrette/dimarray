""" array with physical dimensions (named and valued axes)
"""
import numpy as np
import copy
from collections import OrderedDict

from dimarray.config import Config

from metadata import Metadata
from axes import Axis, Axes, GroupedAxis

import _transform  # numpy along-axis transformations, interpolation
import _reshape	   # change array shape and dimensions
import _indexing   # perform slicing and indexing operations

from dimarray.lib.tools import pandas_obj

__all__ = ["Dimarray", "array"]

class Dimarray(Metadata):
    """ numpy's ndarray with physical dimensions (named and values axes)

    Initialization:
    ---------------

    Dimarray(values)
    Dimarray(values, 'x0', 'x1', ...)
    Dimarray(values, val0, val1, ...)
    Dimarray(values, ('x0', val0), ('x1', val1), ...)
    Dimarray(values, x0=val0, x1=val1, ...)   # ONLY if val0.size != val1.size 
    Dimarray(values, ..., name='mydata', units='m')  

    Parameters:
    -----------
    values	: numpy-like array, or Dimarray instance
    axes	: variable list of Axis objects or list of tuples ('dim', values)

    Optional keyword arguments **kwargs: 

    options: 
	- dtype, copy: passed to np.array()

	Internal use only:

	- _INDEXING: default "index". This is the default value for `method=` in 
		    xs(method=...) and this also determine the behaviour of 
		    __getitem__. Note the `ix` property acts as a toogle 
		    between "numpy" and all other methods (back to `index`)

    metadata: <att=val> keywords, where val can be any mutable object 
	      (`str`, `tuple`, `int`, `float`...).
	      metadata describe the array, including "name", "units" and "stamp"
	      `att` cannot be one of ("dtype", "copy", "INDEXING")
	      In these particular cases, use setncattr method instead.
    
    kwaxes: axes as keyword arguments, <name = values>, where values is 
	    a list or ndarray object

    The latter options is provided as convenience
     
    Note this is similar to Dimarray.from_kw except that it does not 
    allow for providing `dims=`, so that it cannot resolve ambiguous cases
    such as two or more axes with the same size. For such cases, use 
    Dimarray.from_kw or another format for input axes.


    Examples:
    ---------

    Basic:

    >>> Dimarray([[1,2,3],[4,5,6]]) # automatic labelling
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[1, 2, 3],
	   [4, 5, 6]])

    >>> Dimarray(values, 'items','time')  # axis names only
    dimarray: 6 non-null elements (0 null)
    dimensions: 'items', 'time'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[1, 2, 3],
	   [4, 5, 6]])

    >>> Dimarray(values, list("ab"), np.arange(1950,1953)) # axis values only
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 'a' to 'b'
    1 / x1 (3): 1950 to 1953
    array([[1, 2, 3],
	   [4, 5, 6]])

    More general case

    >>> a = Dimarray(values, ('items',list("ab")), ('time',np.arange(1950,1953)))
    >>> b = Dimarray(values, items=list("ab"), time=np.arange(1950,1953))
    >>> c = Dimarray.from_kw(values, items=list("ab"), time=np.arange(1950,1953), dims=['items', 'time']) 
    >>> d = Dimarray.from_list(values, [list("ab"), np.arange(1950,1953)], dims=['items','time']) 
    >>> a == b == c == d
    True
    >>> a
    dimarray: 6 non-null elements (0 null)
    dimensions: 'items', 'time'
    0 / items (2): a to b
    1 / time (3): 1950 to 1952
    array([[1, 2, 3],
	   [4, 5, 6]])
    """
    _order = None  # set a general ordering relationship for dimensions

    _metadata_exclude = ("values","axes") # is NOT a metadata

    #
    # NOW MAIN BODY OF THE CLASS
    #

    def __init__(self, values, *axes, **kwargs):
	""" Initialization
	"""

	# filter **kwargs to retrieve axes
	kwaxes = {}
	for k in kwargs.keys():
	    if type(kwargs[k]) in (list, np.ndarray):
		kwaxes[k] = kwargs.pop(k)

	# define axes from keyword arguments, if applicable
	assert not (len(kwaxes) > 0 and len(axes) > 0), "cannot provide both list and keywords arguments"
	if len(kwaxes) > 0:
	    data = self.__class__.from_kw(values, **kwaxes)
	    axes = data.axes

	# filter **kwargs to retrieve options
	opt = dict(dtype=None, copy=False, _INDEXING=Config.indexing)
	for k in opt:
	    if k in kwargs:
		opt[k] = kwargs.pop(k)


	# what remains is metadata
	metadata = kwargs

	#
	# array values
	#
	avalues = np.array(values, copy=opt['copy'], dtype=opt['dtype'])

	#
	# Initialize the axes
	# 
	# Can be one of
	# - list of Axis objects
	# - list of tuples `dim, array`
	# - list of str for dimension names
	# - list of arrays (unnamed dimensions)
	# - nothing

	# axis not provided: check whether values has an axes field
	if len(axes) == 0:

	    # check if attached to values (e.g. Dimarray or pandas object)
	    if hasattr(values, "axes"):
		axes = values.axes

	    # define a default set of axes if not provided
	    axes = Axes.from_shape(avalues.shape)

	# list of axes
	elif isinstance(axes[0], Axis):
	    axes = Axes(axes)

	# (name, values) tuples
	elif isinstance(axes[0], tuple):
	    axes = Axes.from_tuples(*axes)

	# only axis values are provided: default names
	elif isinstance(axes[0], np.ndarray) or isinstance(axes[0], list):
	    axes = Axes.from_list(axes)

	# only dimension names are provided: default values
	elif isinstance(axes[0], str):
	    axes = Axes.from_shape(avalues.shape, dims=axes)

	else:
	    raise TypeError("axes, if provided, must be a list of: `Axis` or `tuple` or `str` or arrays")

	#
	# store all fields
	#

	# init an ordered dict self._metadata
	super(Dimarray, self).__init__() 

	self.values = avalues
	self.axes = axes

	# options
	self._INDEXING = opt['_INDEXING']

	#
	# metadata (see Metadata type in metadata.py)
	#
	for k in metadata:
	    self.setncattr(k, metadata[k]) # perform type-checking and store in self._metadata

	# Check consistency between axes and values
	inferred = tuple([ax.size for ax in self.axes])
	if inferred != self.values.shape:
	    print "shape inferred from axes: ",inferred
	    print "shape inferred from data: ",self.values.shape
	    raise Exception("mismatch between values and axes")

	# If a general ordering relationship of the class is assumed,
	# always sort the class
	if self._order is not None:
	    present = filter(lambda x: x in self.dims, self._order)  # prescribed
	    missing = filter(lambda x: x not in self._order, self.dims)  # not
	    order = missing + present # prepend dimensions not found in ordering relationship
	    self.transpose(order, inplace=True)

    #
    # Internal constructor, useful for subclassing
    #
    @classmethod
    def _constructor(cls, values, axes, **metadata):
	""" Internal API for the constructor: check whether a pre-defined class exists

	values	: numpy-like array
	axes	: Axes instance 

	This static method is used whenever a new Dimarray needs to be instantiated
	for example after a transformation.

	This makes the sub-classing process easier since only this method needs to be 
	overloaded to make the sub-class a "closed" class.
	"""
	return cls(values, *axes, **metadata)

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
		return super(Dimarray, self).__getattr__(att) # call Metadata's method

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
	#return Dimarray(self.values.copy(), self.axes.copy(), slicing=self.slicing, **{k:getattr(self,k) for k in self.ncattrs()})

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
	    val = self.xs(k, axis=axis) # cross-section
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
	    weights: Dimarray of weights, or None

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

	    if weights == 1:
		return None
	    else:
		weights = weights.values

	# Now create a dimarray and make it full size
	# ...already full-size
	if weights.shape == self.shape:
	    weights = Dimarray(weights, *self.axes)

	# ...need to be expanded
	elif weights.shape == tuple(ax.size for ax in axes):
	    weights = Dimarray(weights, *axes).expand(self.dims)

	else:
	    try:
		weights = weights + np.zeros_like(self.values)

	    except:
		raise ValueError("weight dimensions are not conform")
	    weights = Dimarray(weights, *self.axes)

	#else:
	#    raise ValueError("weight dimensions not conform")

	# fill NaNs in when necessary
	if fill_nans:
	    weights.values[np.isnan(self.values)] = np.nan
	
	return weights


    #
    # INDEXING
    #

    #
    # New general-purpose indexing method
    #
    xs = _indexing.xs 

    #
    # Standard indexing takes axis values
    #
    def __getitem__(self, item): 
	""" get a slice (use xs method)
	"""
	items = np.index_exp[item] # tuple
    
	# dictionary <axis name> : <axis index> to feed into xs
	ix_nd = {self.axes[i].name: it for i, it in enumerate(items)}

	return self.xs(**ix_nd)

    # 
    # Can also use integer-indexing via ix
    #
    @property
    def ix(self):
	""" integer index-access (unless _INDEXING is already numpy ==> go to index)
	"""
	# special case where user set self.INDEXING to "index"
	if self._INDEXING == "numpy":
	    _INDEXING = "index"

	# standard case:
	else:
	    _INDEXING = "numpy"

	return self._constructor(self.values, self.axes, _INDEXING=_INDEXING, **self._metadata)

    #
    # TRANSFORMS
    # 

    #
    # ELEMENT-WISE TRANSFORMS
    #
    def apply(self, func, args=(), **kwargs):
	""" Apply element-wise function to Dimarray
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
    
    @property
    def T(self):
	return self.transpose()


    #
    # REINDEXING 
    #
    reindex_axis = _indexing.reindex_axis
    reindex_like = _indexing.reindex_like

    #
    # BASIC OPERATTIONS
    #
    def _operation(self, func, other):
	""" make an operation: this include axis and dimensions alignment

	Just for testing:
	>>> b
	...
	array([[ 0.,  1.],
	       [ 1.,  2.]])
	>>> b == b
	True
	>>> b+2 == b + np.ones(b.shape)*2
	True
	>>> b+b == b*2
	True
	>>> b*b == b**2
	True
	>>> (b - b.values) == b - b
	True
	"""
	result = _operation(func, self, other, broadcast=Config.op_broadcast, reindex=Config.op_reindex, constructor=self._constructor)
	return result

    def __add__(self, other): return self._operation(np.add, other)

    def __sub__(self, other): return self._operation(np.subtract, other)

    def __mul__(self, other): return self._operation(np.multiply, other)

    def __div__(self, other): return self._operation(np.divide, other)

    def __pow__(self, other): return self._operation(np.power, other)

    def __float__(self):  return float(self.values)
    def __int__(self):  return int(self.values)

    def __eq__(self, other): 
	#return isinstance(other, Dimarray) and np.all(self.values == other.values) and self.axes == other.axes
	if not isinstance(other, Dimarray) or not self.axes == other.axes:
	    return False

	return self._constructor(self.values == other.values, self.axes)

    ##
    ## to behave like a dictionary w.r.t. first dimension (pandas-like)
    ##
    #def __iter__(self):
    #    for k in self.keys():
    #        yield k

    #def keys(self):
    #    return self.axes[0].values

    #
    # IMPORT FROM / EXPORT TO
    #
    @classmethod
    def from_list(cls, values, axes=None, dims=None, **kwargs):
	""" initialize Dimarray with with variable list of tuples, and attributes

	values	    : numpy-like array (passed to array())
	axes	    : list of axis values
	dims	    : list of axis dims
	(axes, dims) are passed to Axes.from_list
	**kwargs    : passed to Dimarray (attributes)

	Example:
	--------
	"""
	if axes is None:
	    axes = Axes.from_shape(np.shape(values), dims)

	else:
	    axes = Axes.from_list(axes, dims)

	return cls(values, *axes, **kwargs)

    @classmethod
    def from_kw(cls, values, dims=None, **kwaxes):
	""" initialize Dimarray with with axes as key-word arguments

	values	    : numpy-like array (passed to array())
	**kwaxes      : axes passed as keyword arguments <axis name>=<axis val>

	NOTE: no attribute can be passed with the from_kw method 
	      ==> try using setncattr() method instead or for example:
	      a = Dimarray.from_kw(values, **kwaxes)
	      a.name = "myname"
	      a.setncattr('units', "myunits")
	
	Examples:
	---------
	>>> Dimarray.from_kw([[1,2,3],[4,5,6]], items=list("ab"), time=np.arange(1950,1953))
	dimarray: 6 non-null elements (0 null)
	dimensions: 'items', 'time'
	0 / items (2): a to b
	1 / time (3): 1950 to 1952
	array([[1, 2, 3],
	       [4, 5, 6]])

	>>> Dimarray.from_kw([[1,2],[4,5]], items=list("ab"), time=np.arange(1950,1952), dims=['items','time'])
	dimarray: 4 non-null elements (0 null)
	dimensions: 'items', 'time'
	0 / items (2): a to b
	1 / time (2): 1950 to 1951
	array([[1, 2],
	       [4, 5]])
	"""
	# "convenience", quick check on axes, to avoid a "deep" error message
	for k in kwaxes:
	    if type(kwaxes[k]) is str:
		print "PASSED:",k,":",kwaxes[k]
		raise ValueError("invalid axis values, see doc about how to pass metadata")

	list_axes = Axes.from_kw(shape=np.shape(values), dims=dims, **kwaxes)

	return cls._constructor(values, list_axes)

    @classmethod
    def from_dataset(cls, data, keys=None, axis="items"):
	""" aggregate a collection of N-D Dimarrays into a N+1 D dimarray

	Convenience method for: collect.Dataset(data, keys).to_array(axis)

	input:
	    - data : list or dict of Dimarrays
	    - keys, optional : ordering of data (for dict)
	    - axis, optional : axis along which to aggregate data (default "items")

	output:
	    - new Dimarray object
	"""
	from dataset import Dataset
	data = Dataset(data, keys=keys)
	return data.to_array(axis=axis)

    def to_dataset(self, axis=0):
	""" split a Dimarray into a Dataset object (collection of Dimarrays)
	"""
	from dataset import Dataset
	# iterate over elements of one axis
	data = [val for k, val in self.iter(axis)]
	return Dataset(data)



    #
    # export to other data types
    #
    def to_MaskedArray(self):
	""" transform to MaskedArray, with NaNs as missing values
	"""
	return _transform.to_MaskedArray(self.values)

    def to_list(self):
	return [self[k] for k in self]

    def to_dict(self, axis=0):
	""" return an ordered dictionary of sets
	"""
	from dataset import Dataset
	d = dict()
	for k, v in self.iter(axis):
	    d[k] = v
	return Dataset(d, keys=self.axes[axis].values)

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

	# show metadata as well?
	if len(self.ncattrs()) > 0:
	    line = self.repr_meta()
	    lines.append(line)

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

    def write(self, f, name=None, *args, **kwargs):
	import dimarray.io.nc as ncio

	# add variable name if provided...
	if name is None and hasattr(self, "name"):
	    name = self.name

	ncio.write_variable(f, self, name, *args, **kwargs)

    @classmethod
    def read(cls, f, *args, **kwargs):
	import dimarray.io.nc as ncio
	return ncio.read_base(f, *args, **kwargs)

    #
    # Plotting
    #
    def plot(self, *args, **kwargs):
	""" by default, use pandas for plotting
	"""
	assert self.ndim <= 2, "only support plotting for 1- and 2-D objects"
	return self.to_pandas().plot(*args, **kwargs)


def array(values, *args, **kwargs):
    """ alias for Dimarray
    """
    return Dimarray(values, *args, **kwargs)

array.__doc__ += Dimarray.__doc__.replace("Dimarray","da.array")

