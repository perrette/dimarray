""" Array with dimensions
"""
import numpy as np
import copy
from collections import OrderedDict
#import json

from metadata import Metadata
from axes import Axis, Axes, GroupedAxis

import _transform  # numpy along-axis transformations, interpolation
import _reindex	   # re-index + interpolation
import _reshape	   # change array shape and dimensions
import _indexing   # perform slicing and indexing operations

from tools import pandas_obj
#import plotting

__all__ = []

#
# Main class:
#
class Dimarray(Metadata):
    """ numpy's ndarray with physical dimensions

    * numpy's ndarrays with named axes and metadata organized on the netCDF format.
    * follow pandas' api but works for any dimension and any axis name
    * includes most numpy transformations
    * alias method for matplotlib plotting.
    * natural netCDF I/O
    """
    _order = None  # set a general ordering relationship for dimensions

    _metadata_exclude = ("values","axes") # is NOT a metadata

    #
    # NOW MAIN BODY OF THE CLASS
    #

    def __init__(self, values, *axes, **kwargs):
	""" Initialization

	values	: numpy-like array, or Dimarray instance
	axes	: variable list of Axis objects or list of tuples ('dim', values)
		  This argument can only be omitted if values is an instance of Dimarray

	options:

	    - dtype, copy: passed to np.array()

	key-word arguments:
	    - metadata
	"""
	# filter **kwargs and keep metadata
	default = dict(dtype=None, copy=False, _indexing="index")
	for k in default:
	    if k in kwargs:
		default[k] = kwargs.pop(k)

	metadata = kwargs
	_indexing = default.pop('_indexing')

	#
	# array values
	#
	avalues = np.array(values, **default)

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
	self._indexing = _indexing

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
	    return super(Dimarray, self).__getattr__(att) # call Metadata's method

    def set(self, inplace=False, **kwargs):
	""" update multiple class attributes in-place or after copy

	inplace: modify attributes in-place, return None 
	otherwise first make a copy, and return new obj

	a.set(_indexing="numpy")[:30]
	a.set(_indexing="index")[1971.42]
	a.set(_indexing="nearest")[1971]
	a.set(name="myname", inplace=True) # modify attributes inplace
	"""
	if inplace: 
	    for k in kwargs:
		setattr(self, k, kwargs[k]) # include metadata check
	    #self.__dict__.update(kwargs)

	else:
	    obj = self.copy(shallow=True)
	    for k in kwargs:
		setattr(obj, k, kwargs[k]) # include metadata check
	    return obj

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
	""" integer index-access
	"""
	return self._constructor(self.values, self.axes, _indexing="numpy", **self._metadata)

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
    median = _transform.median
    sum = _transform.sum

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
    # OTHER transformations 
    #

    # Recursive application of obj => obj transformation
    apply_recursive = _transform.apply_recursive 

    # 1D AND BILINEAR INTERPOLATION (RECURSIVELY APPLIED)
    interp1d = _transform.interp1d_numpy
    interp2d = _transform.interp2d_mpl


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
    reindex_axis = index_tricks.reindex_axis

    #
    # BASIC OPERATTIONS
    #
    def _operation(self, func, other, reindex=True, transpose=True):
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
	result = _operation(func, self, other, constructor=self._constructor)
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
	**kwaxes      : axes passed as keyword araguments <axis name>=<axis val>
	"""
	# "convenience", quick check on axes, to avoid a "deep" error message
	for k in kwaxes:
	    if type(kwaxes[k]) is str:
		print "PASSED:",k,":",kwaxes[k]
		msg = \
""" no attribute can be passed with the from_kw method 
==> try using the set() method instead, for example:
a = Dimarray.from_kw(values, **kwaxes)
a.name = "myname"
a.units = "myunits"
"""
		raise ValueError(msg)

	list_axes = Axes.from_kw(shape=np.shape(values), dims=dims, **kwaxes)

	return cls(values, *list_axes)

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
	from collect import Dataset
	data = Dataset(data, keys=keys)
	return data.to_array(axis=axis)

    def to_dataset(self, axis=0):
	""" split a Dimarray into a Dataset object (collection of Dimarrays)
	"""
	from collect import Dataset
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
	from collect import Dataset
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

	if self.size < 100:
	    line = repr(self.values)
	    lines.append(line)

	return "\n".join(lines)


    #
    #  I/O
    # 

    def write(self, f, name=None, *args, **kwargs):
	import ncio

	# add variable name if provided...
	if name is None and hasattr(self, "name"):
	    name = self.name

	ncio.write_variable(f, self, name, *args, **kwargs)

    @classmethod
    def read(cls, f, *args, **kwargs):
	import ncio
	return ncio.read_base(f, *args, **kwargs)

    #
    # Plotting
    #
    def plot(self, *args, **kwargs):
	""" by default, use pandas for plotting
	"""
	assert self.ndim <= 2, "only support plotting for 1- and 2-D objects"
	return self.to_pandas().plot(*args, **kwargs)


def array(values, axes=None, dims=None, **kwaxes):
    """ Wrapper for initialization

    a wrapper to Dimarray.from_keys and Dimarray.from_list
    but accepting no metadata.

    a = array(values, lon=mylon, lat=mylat)
    a.set(name="myname", units="units")
    """
    if axes is not None or len(kwaxes) == 0:
	assert len(kwaxes) == 0, "cannot provide both axes and kwaxes, use Dimarray.from_list"
	new = Dimarray.from_list(values, axes, dims=dims) 

    else:
	new = Dimarray.from_kw(values, dims, **kwaxes) 


    return new

#
# Operation and axis aligmnent
#


def _operation(func, o1, o2, reindex=True, transpose=True, constructor=Dimarray):
    """ operation on LaxArray objects

    input:
	func	: operator
	o1    	: LHS operand: Dimarray
	o2    	: RHS operand: at least: be convertible by np.array())
	align, optional: if True, use pandas to align the axes

    output:
	values: array values
	dims : dimension names
    """
    # second operand is not a Dimarray: let numpy do the job 
    if not isinstance(o2, Dimarray):
	if np.ndim(o2) > o1.ndim:
	    raise ValueError("bad input: second operand's dimensions not documented")
	res = func(o1.values, np.array(o2))
	return constructor(res, o1.axes)

    # both objects are dimarrays

    # re-index 
    if reindex:
	o1, o2 = align_axes(o1, o2)
	## common list of dimensions
	#dims1 =  [ax.name for ax in o1.axes] 
	#common_dims =  [ax.name for ax in o2.axes if ax.name in dims1] 

	## reindex both operands
	#for name in common_dims:
	#    ax_values = _common_axis(o1.axes[name].values, o2.axes[name].values)
	#    o1 = o1.reindex_axis(ax_values, axis=name)
	#    o2 = o2.reindex_axis(ax_values, axis=name)

    # determine the dimensions of the result
    newdims = _get_dims(o1, o2) 

    # make sure all dimensions are present
    o1 = o1.reshape(newdims)
    o2 = o2.reshape(newdims)
    assert o1.dims == o2.dims, "problem in transpose"

    # make the new axes
    newaxes = o1.axes.copy()
    # ...make sure no singletong value is included
    for i, ax in enumerate(newaxes):
	if ax.values[0] is None:
	    newaxes[i] = o2.axes[ax.name]

    res = func(o1.values, o2.values)

    return constructor(res, newaxes)

#
# Axis alignment for operations
#
def _common_axis(*axes):
    """ find the common axis between 
    """
    #from heapq import merge
    import itertools

    # First merge the axes with duplicates (while preserving the order of the lists)
    axes_lists = [list(ax.values) for ax in axes] # axes as lists
    newaxis_val = axes_lists[0]
    for val in itertools.chain(*axes_lists[1:]):
	if val not in newaxis_val:
	    newaxis_val.append(val)

    return Axis(newaxis_val, axes[0].name)

def _get_dims(*objects):
    """ find all dimensions from a variable list of objects
    """
    dims = []
    for o in objects:
	for dim in o.dims:
	    if dim not in dims:
		dims.append(dim)

    return dims


def align_axes(*objects):
    """ align axes of a list of objects by reindexing
    """
    # find the dimensiosn
    dims = _get_dims(*objects)

    objects = list(objects)
    for d in dims:

	# objects which have that dimension
	objs = filter(lambda o: d in o.dims, objects)

	# common axis to reindex on
	ax_values = _common_axis(*[o.axes[d] for o in objs])

	# update objects
	for i, o in enumerate(objects):
	    if o not in objs:
		continue
	    if o.axes[d] == ax_values:
		continue

	    objects[i] = o.reindex_axis(ax_values, axis=d)

    return objects

def _ndindex(indices, axis_id):
    """ return the N-D index from an along-axis index
    """
    return (slice(None),)*axis_id + np.index_exp[indices]
