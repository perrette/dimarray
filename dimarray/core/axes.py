import numpy as np
from collections import OrderedDict as odict
import string
import copy

from metadata import MetadataDesc
from tools import is_DimArray
from tools import is_array1d_equiv

__all__ = ["Axis","Axes", "is_regular"]

def is_regular(values):
    """ test if numeric, monotonically increasing and constant step
    """
    if values.dtype is np.dtype('O'): 
	regular = False

    else:
	diff = np.diff(values)
	step = diff[0]
	regular = np.all(diff==step) and step > 0

    return regular

def is_monotonic(values):
    """ test is monotonically increasing or decreasing
    """
    if values.size < 2:
	monotonic = True

    else:
	#increasing = np.diff(values) > 0 
	increasing = values[1:] >= values[:-1] 
	monotonic = np.all(increasing) or np.all(values[1:] <= values[:-1])

    return monotonic


def _convert_dtype(dtype):
    """ convert numpy type in a python type
    """
    if dtype is np.dtype(int):
	type_ = int

    # objects represent strings
    elif dtype is np.dtype('O'):
	type_ = str

    else:
	type_ = float

    return type_

#
# Axis class
#
class Axis(object):
    """ Axis

    values: numpy array (or list) 
    name  : name (attribute)

    weights: [None] associated list of weights 
    modulo: [None] if not None, consider axis values as being defined +/- n*modulo, where n is an integer
	    this modify `loc` behaviour only, but does not impose any constraint on actual axis values

    + metadata
    """
    _metadata = MetadataDesc(exclude = ["values", "name", "weights", "modulo"]) # variables which are NOT metadata

    def __init__(self, values, name="", weights=None, modulo=None, dtype=None, _monotonic=None, **kwargs):
	""" 
	"""
	if not name:
	    assert hasattr(values, "name"), "unnamed dimension !"
	    name = values.name # e.g pandas axis

	#if np.size(values) == 0:
	#    raise ValueError("cannot define an empty axis")
	if np.isscalar(values):
	    raise TypeError("an axis cannot be a scalar value !")

	values = np.asarray(values, dtype)

	# Treat the particular case of a sequence of sequences, leads to a 2-D array
	# ==> convert to a list of tuples
	if values.ndim == 2: 
	    val = np.empty(values.shape[0], dtype=object)
	    val[:] = zip(*values.T.tolist()) # pass a list of tuples
	    values = val

	# check
	if values.ndim != 1:
	    raise ValueError("an Axis object can only be 1-D, check-out GroupedAxis")

	# convert strings to object type
	if values.dtype not in (np.dtype(float), np.dtype(int), np.dtype(long)):
	    values = np.asarray(values, dtype=object)

	self.values = values 
	self.name = name 
	self.weights = weights 
	self.modulo = modulo
	self._monotonic = _monotonic

	self._metadata = kwargs

    def __getitem__(self, item):
	""" access values elements & return an axis object
	"""
	if type(item) is slice and item == slice(None):
	    return self

	values = self.values[item]

	# if collapsed to scalar, just return it
	if not isinstance(values, np.ndarray):
	    return values

	if isinstance(self.weights, np.ndarray):
	    weights = self.weights[item]

	else:
	    weights = self.weights

	return Axis(values, self.name, weights=weights, **self._metadata)

    def union(self, other):
	""" join two Axis objects

	Examples:
	---------

	>>> ax1 = Axis([0, 1, 2, 3, 4], name='myaxis')
	>>> ax2 = Axis([-3, 2, 3, 6], name='myaxis')
	>>> ax3 = ax1.union(ax2)
	>>> ax3.values
	array([-3,  0,  1,  2,  3,  4,  6])
	"""
	#assert isinstance(other, Axis), "can only make the Union of two Axis objects"
	if isinstance(other, Axis):
	    assert self.name == other.name, "axes have different names, cannot make union"
	else:
	    other = Axis(other, self.name) # to give it the same methods is_monotonic etc...
	
	if np.all(self.values == other.values):
	    # TODO: check other attributes such as weights
	    return self.copy()

	joined = np.unique(np.concatenate([self.values, [val for val in other.values if val not in self.values]]))

	# join two sorted axes?
	if self.is_monotonic() and other.is_monotonic():
	    is_increasing = lambda ax: ax.size < 2 or ax.values[1] >= ax.values[0]
	    is_decreasing = lambda ax: ax.size < 2 or ax.values[1] <= ax.values[0]

	    if is_increasing(self) and is_increasing(other):
		joined.sort()

	    elif is_decreasing(self) and is_decreasing(other):
		joined2 = joined[::-1] # reverse should be increasing
		joined2.sort()
		joined = joined2[::-1] # back to normal

	    # one increases, the other decreases !
	    else:
		joined.sort()

	return Axis(joined, self.name)

    def is_monotonic(self):
	""" return True if monotonic
	"""
	if self._monotonic is None:
	    self._monotonic = is_monotonic(self.values)

	return self._monotonic

    def is_regular(self):
	""" return True if regular axis (numeric and steadily increasing)
	"""
	if self._regular is None:
	    self._regular = self.is_numeric() and self.is_monotonic() and is_regular(self.values)

	return self._regular

    @property
    def weights(self):
	return self._weights

    @weights.setter
    def weights(self, _weights):
	if _weights is None or hasattr(_weights, '__call__'):
	    pass

	else:
	    try:
		_weights = np.asarray(_weights)
	    except:
		raise TypeError("weight mut be array-like or callable, got: {}".format(_weights))

	    if _weights.size != self.values.size:
		raise ValueError("weights must have the same size as axis values, got: {} and {} !".format(_weights.size, self.values.size))

	self._weights = _weights

    @property
    def loc(self):
	""" Access the slicer to locate axis elements

	>>> ax = Axis([1,2,3],'x0')
	>>> ax.is_numeric()
	True
	>>> ax = Axis(['a','b','c'],'x0')
	>>> ax.is_numeric()
	False
	"""
	assert self.values.ndim == 1, "!!! 2-dimensional axis !!!"
	if self.is_numeric():
	    return NumLocator(self.values, modulo=self.modulo)
	else:
	    return ObjLocator(self.values)

    def is_numeric(self):
	""" numeric type?
	"""
	return self.values.dtype in (np.dtype(int), np.dtype(long), np.dtype(float))
	#try:
	#    self.values[0] + 1
	#    return True
	#except:
	#    return False

    def __eq__(self, other):
	#return hasattr(other, "name") and hasattr(other, "values") and np.all(other.values == self.values) and self.name == other.name
	return isinstance(other, Axis) and np.all(other.values == self.values) and self.name == other.name

    def __repr__(self):
	""" string representation for printing to screen
	""" 
	return "{} ({}): {} to {}".format(self.name, self.size, *self._bounds())

    def _bounds(self):
	if self.values.size == 0:
	    start, stop = None, None
	else:
	    start, stop = self.values[0], self.values[-1]
	return start, stop

    def __str__(self):
	""" simple string representation
	"""
	#return "{}={}:{}".format(self.name, self.values[0], self.values[-1])
	return "{}({})={}:{}".format(self.name, self.size, *self._bounds())

    def copy(self):
	return copy.copy(self)

    # a few array-like properties
    @property
    def size(self): 
	return self.values.size

    @property
    def dtype(self): 
	return _convert_dtype(np.array(self.values).dtype)

    @property
    def __array__(self): 
	return self.values.__array__

    @property
    def __len__(self): 
	return self.values.__len__

    def get_weights(self, weights=None):
	""" return axis weights as a DimArray
	"""
	from dimarraycls import DimArray

	if weights is None:
	    weights = self.weights

	# no weights
	if weights is None:
	    weights = np.ones_like(self.values)

	# function of axis
	elif callable(weights):
	    weights = weights(self.values)

	# same weight for every element
	elif np.size(weights) == 1:
	    weights = np.zeros_like(self.values) + weights

	# already an array of weights
	else:
	    weights = np.array(weights, copy=False)

	# index on one dimension
	ax = Axis(self.values, name=self.name)

	return DimArray(weights, ax)

    def to_pandas(self):
	""" convert to pandas Index
	"""
	import pandas as pd
	return pd.Index(self.values, name=self.name)

    @classmethod
    def from_pandas(cls, index):
	return cls(index.values, name=index.name)

class GroupedAxis(Axis):
    """ an Axis which is a grouping of several axes flattened together
    """
    modulo = None

    _metadata = MetadataDesc(exclude = ["values", "name", "weights", "modulo","axes"]) # variables which are NOT metadata

    def __init__(self, *axes):
	"""
	"""
	self.axes = Axes(axes)
	self.name = ",".join([ax.name for ax in self.axes])
	self._values = None  # values not computed unless needed
	self._weights = None  
	self._size = None  

    @property
    def values(self):
	""" values as 2-D numpy array, to keep things consistent with Axis
	"""
	if self._values is None:
	    self._values = self._get_values()
	return self._values

    def _get_values(self):
	# Each element of the new axis is a tuple, which makes a 2-D numpy array
	aval = _flatten(*[ax.values for ax in self.axes])
	val = np.empty(aval.shape[0], dtype=object)
	val[:] = zip(*aval.T.tolist()) # pass a list of tuples
	return val 

    @property
    def weights(self):
	""" Combine the weights as a product of the individual axis weights
	"""
	if self._weights is None:
	    self._weights = self._get_weights()
	return self._weights

    def _get_weights(self):
	if np.all([ax.weights is None for ax in self.axes]):
	    return None
	else:
	    return _flatten(*[ax.get_weights() for ax in self.axes]).prod(axis=1)

    @property
    def size(self): 
	""" size as product of axis sizes
	"""
	if self._size is None:
	    self._size = self._get_size()
	return self._size

    def _get_size(self):
	return np.prod([ax.size for ax in self.axes])

    @property
    def levels(self):
	""" for convenience, return individual axis values (like pandas)
	"""
	return [ax.values for ax in self.axes]

    def __repr__(self):
	""" string representation
	""" 
	first, last = zip(*[ax._bounds() for ax in self.axes])
	return "{} ({}): {} to {}".format(self.name, self.size, first, last)

    def __str__(self):
	return ",".join([str(ax) for ax in self.axes])

    def to_MultiIndex(self):
	""" convert to pandas MultiIndex
	"""
	import pandas as pd
	index = pd.MultiIndex.from_tuples(self.values, names=[ax.name for ax in self.axes])
	index.name = self.name
	return index

    to_pandas = to_MultiIndex # over load Axis method

    @classmethod
    def from_MultiIndex(cls, mi):
	""" from a pandas MultiIndex
	"""
	axes = []
	for i, lev in enumerate(mi.levels):
	    nm = lev.name
	    if nm is None: 
		nm = "lev{}".format(i)
	    axes.append(Axis(lev.values, nm))

	ax = cls(*axes)
	if mi.name is not None:
	    ax.name = mi.name

	return ax

def _flatten(*list_of_arrays):
    """ flatten a list of arrays ax1, ax2, ... to  a list of tuples [(ax1[0], ax2[0], ax3[0]..), (ax1[0], ax2[0], ax3[1]..), ...]
    """
    assert len(list_of_arrays) > 1, "only one axis!"

    kwargs = dict(indexing="ij")
    grd = np.meshgrid(*list_of_arrays, **kwargs)
    array_of_tuples = np.array(zip(*[g.ravel() for g in grd]))
    assert array_of_tuples.shape[1] == len(list_of_arrays), "pb when reshaping: {} and {}".format(array_of_tuples.shape, len(list_of_arrays))
    assert array_of_tuples.shape[0] == np.prod([x.size for x in list_of_arrays]), "pb when reshaping: {} and {}".format(array_of_tuples.shape, np.prod([x.size for x in list_of_arrays]))
    return array_of_tuples

#
# List of axes
#

class Axes(list):
    """ Axes class: inheritates from a list but dict-like access methods for convenience
    """
    def __init__(self, *args, **kwargs):

	list.__init__(self, *args, **kwargs)
	for v in self:
	    if not isinstance(v, Axis):
		raise TypeError("an Axes object can only be initialized with a list of Axes objects, got: {} (instance:{}) !".format(type(v), v))

    def append(self, item):
	""" add a check on axis
	"""
	# if item is an Axis, just append it
	assert isinstance(item, Axis), "can only append an Axis object !"
	#super(Axes, self).append(item)
	list.append(self, item)

    @staticmethod
    def _init(*args, **kwargs):
	# try to catch errors one level higher
	try:
	    axes = _init_axes(*args, **kwargs)
	except TypeError, msg:
	    raise TypeError(msg)
	except ValueError, msg:
	    raise ValueError(msg)
	return axes

    @classmethod
    def from_tuples(cls, *tuples_name_values):
	""" initialize axes from tuples

	Axes.from_tuples(('lat',mylat), ('lon',mylon)) 
	"""
	assert np.all([type(tup) is tuple for tup in tuples_name_values]), "need to provide a list of `name, values` tuples !"

	newaxes = cls()
	for nm, values in tuples_name_values:
	    newaxes.append(Axis(values, nm))
	return newaxes

    @classmethod
    def from_shape(cls, shape, dims=None):
	""" return default axes based on shape
	"""
	axes = cls()
	for i,ni in enumerate(shape):
	    if dims is None:
		name = "x{}".format(i) # default name
	    else:
		name = dims[i]
	    axis = Axis(np.arange(ni), name)
	    axes.append(axis)

	return axes

    @classmethod
    def from_arrays(cls, arrays, dims=None):
	"""  list of np.ndarrays and dims
	"""
	assert np.iterable(arrays) and (dims is None or len(dims) == len(arrays)), "invalid input arrays={}, dims={}".format(arrays, dims)

	# default names
	if dims is None: 
	    dims = ["x{}".format(i) for i in range(len(arrays))]

	return cls.from_tuples(*zip(dims, arrays))

    @classmethod
    def from_dict(cls, kwaxes, dims=None, shape=None, raise_warning=True):
	""" infer dimensions from key-word arguments
	"""
	# if no key-word argument is given, just return default axis
	if len(kwaxes) == 0:
	    return cls.from_shape(shape, dims)

	axes = cls()
	for k in kwaxes:
	    axes.append(Axis(kwaxes[k], k))

	# Make sure the order is right (since it is lost via dict-passing)

	# preferred solution: dims is given
	if dims is not None:
	    axes.sort(dims)

	# alternative option: only the shape is given
	elif shape is not None:
	    assert len(shape) == len(kwaxes), "shape does not match kwaxes !"
	    current_shape = [ax.size for ax in axes]
	    assert set(shape) == set(current_shape), "mismatch between array shape and axes"
	    assert len(set(shape)) == len(set(current_shape)) == len(set([ax.name for ax in axes])), \
    """ some axes have the same size !
    ==> ambiguous determination of dimensions order via keyword arguments only
    ==> explictly supply `dims=` or use from_arrays() or from_tuples() methods" """
	    argsort = [current_shape.index(k) for k in shape]

	    assert len(argsort) == len(axes), "keyword arguments do not match shape !"
	    axes = Axes([axes[i] for i in argsort])

	    current_shape = tuple([ax.size for ax in axes])
	    assert current_shape == shape, "dimensions mismatch (axes shape: {} != values shape: {}".format(current_shape, shape)

	elif raise_warning:
	    raise Warning("no shape information: random order")

	dims = [ax.name for ax in axes]
	assert len(set(dims)) == len(dims), "what's wrong??"

	return axes

    def __getitem__(self, k):
	""" get an axis by integer or name
	"""
	k = self.get_idx(k)

	return list.__getitem__(self, k)
	#return super(Axes,self)[item]

    def __setitem__(self, k, item):
	""" get an axis by integer or name
	"""
	k = self.get_idx(k)
	if not isinstance(item, Axis):
	    raise TypeError("can only set axis type, got: {}".format(item))

	return list.__setitem__(self, k, item)

    def __repr__(self):
	""" string representation
	"""
	#header = "dimensions: "+ " x ".join([repr(ax.name) for ax in self])
	header = "dimensions: "+ ", ".join([repr(ax.name) for ax in self])
	body = "\n".join(["{} / {}".format(i, repr(ax).split('\n')[0]) for i,ax in enumerate(self)])
	return "\n".join([header, body])

    def sort(self, dims):
	""" sort IN PLACE according to the order in "dims"
	"""
	if type(dims[0]) is int:
	    dims = [ax.name for ax in self]

	#list.sort(self, key=lambda x: dims.index(x.name))
	super(Axes, self).sort(key=lambda x: dims.index(x.name))

    def copy(self):
	return copy.copy(self)

    def get_idx(self, axis):
	""" always return axis integer location
	"""
	# if axis is already an integer, just return it
	if type(axis) in (int, np.int_, np.dtype(int)):
	    return axis

	assert type(axis) in [str, unicode, tuple, np.string_], "unexpected axis index: {}, {}".format(type(axis), axis)

	dims = [ax.name for ax in self]

	return dims.index(axis)

    @property
    def loc(self):
	return LocatorAxes(self)


def _init_axes(axes=None, dims=None, labels=None, shape=None, raise_warning=True):
    """ initialize axis instance with many different ways

    axes:
	- dict
	- list of Axis objects
	- list of tuples `dim, array`
	- list of arrays, to be complemented by "dims="
	- nothing

    dims: tuple or list of dimension names
    shape
    """
    #back compat
    if axes is None:
	axes = labels 

    # special case: 1D object: accept single axis instead of list of axes/dimensions
    if shape is not None and len(shape) == 1:
	
	# accept a tuple ('dim', axis values) instead of [(...)]
	if type(axes) is tuple:
	    if len(axes) == 2 and type(axes[0]) is str and is_array1d_equiv(axes[1]):
		axes = [axes]

	# accept axes=axis, dims='dim' (instead of list)
	elif (axes is None or is_array1d_equiv(axes)) and (type(dims) in (str, type(None))):
	    #if axes is not None: assert np.size(axes) == values.size, "invalid argument: bad size"
	    axes = [axes] if axes is not None else None
	    dims = [dims] if dims is not None else None

    # axis not provided: check whether values has an axes field
    if axes is None:
	assert shape is not None, "at least shape must be provided (if axes are not)"

	# define a default set of axes if not provided
	axes = Axes.from_shape(shape, dims=dims)
	return axes

    elif isinstance(axes, dict):
	kwaxes = axes
	if isinstance(kwaxes, odict) and dims is None:
	    dims = kwaxes.keys()
	axes = Axes.from_dict(kwaxes, dims=dims, shape=shape, raise_warning=raise_warning)
	return axes

    else:
	if not isinstance(axes, list) and not isinstance(axes, tuple):
	    raise TypeError("axes, if provided, must be a list of: `Axis` or `tuple` or arrays. Got: {} (instance:{})".format(axes.__class__, axes))

    # FROM HERE list or tuple

    # empty axis
    if len(axes) == 0:
	return Axes()

    # list of Axis objects
    elif np.all([isinstance(ax, Axis) for ax in axes]):
	axes = Axes(axes)

    # (name, values) tuples
    elif np.all([isinstance(ax, tuple) for ax in axes]):
	axes = Axes.from_tuples(*axes)

    # axes contains only axis values, with names possibly provided in `dims=`
    elif np.all([type(ax) in (list, np.ndarray) for ax in axes]):
	axes = Axes.from_arrays(axes, dims=dims)

    # axes only cointain axis labels
    elif np.all([type(ax) in (str, unicode) for ax in axes]):
	axes = Axes.from_shape(shape, dims=axes)

    else:
	raise TypeError("axes, if provided, must be a list of: `Axis` or `tuple` or arrays. Got: {} (instance:{})".format(axes.__class__, axes))

    return axes



#
# Locate values on an axis
#

## indexing errors
#class OutBoundError(IndexError):
#    pass

def locate(values, *args, **kwargs):
    return Axis(values).loc(*args, **kwargs)

class LocatorAxis(object):
    """ This class is the core of indexing in dimarray. 

	loc = LocatorAxis(values, **opt)  

    where `values` represent the axis values


    A locator instance is generated from within the Axis object, via 
    its properties loc (valued-based indexing) and iloc (integer-based)

	axis.loc  ==> LocatorAxis(values)  

    A locator is hashable is a similar way to a numpy array, but also 
    callable to update parameters on-the-fly.

    It returns an integer index or `list` of `int` or `slice` of `int` which 
    is understood by numpy's arrays. In particular we have:

	loc[ix] == np.index_exp[loc[ix]][0]

    The "set" method can also be useful for chained calls. We have the general 
    equivalence:

	loc(idx, **kwargs) :: loc.set(**kwargs)[idx]

    """
    _check_params = False # false  for multi indexing
    def __init__(self, values, raise_error=True, position_index = False, keepdims = False, **opt):
	"""
	values	: string list or numpy array

	raise_error = True # raise an error if value not found?
	"""
	# compatibility wiht other methods:
	if 'indexing' in opt:
	    indexing = opt.pop('indexing')
	    assert indexing in ('values', 'position')
	    position_index = indexing == 'position'

	self.values = values
	self.raise_error = raise_error
	self.position_index = position_index
	self.keepdims = keepdims 

	# check parameter values (default to False)
#	if self._check_params:
	for k in opt: 
	    if not hasattr(self, k):
		if k in ('tol', 'modulo'): # need to clean that up in LocatorAxes
		    pass
		else:
		    raise ValueError("unknown parameter {} for {}".format(k, self.__class__))

	assert not hasattr(self, 'indexing')

	#self.__dict__.update(opt) # update default options

    #
    # wrapper mode: __getitem__ and __call__
    #
    def __getitem__(self, ix):
	""" 
	"""
	#
	# check special cases
	#
	assert ix is not None, "index is None!"

	if self.position_index:
	    return ix

	# boolean indexing ?
	if is_DimArray(ix):
	    ix = ix.values

	if type(ix) in (np.ndarray,) and ix.dtype is np.dtype(bool):
	    return ix

	# make sure (1,) is understood as 1 just as numpy would
	elif type(ix) is tuple:
	    if len(ix) == 1:
		ix = ix[0]
	#    else:
	#	raise TypeError("index not understood: did you mean a `slice`?")

	#
	# look up corresponding numpy indices
	#
	# e.g. 45:56
	if type(ix) is slice:
	    res = self.slice(ix)

	elif self._islist(ix):
	    res = map(self.locate, ix)

	else:
	    res = self.locate(ix)

	return res

    def _islist(self, ix):
	""" check if value is a list index (in the sense it will collapse an axis)
	"""
	return type(ix) in (list, np.ndarray)

    def __call__(self, ix, **kwargs):
	""" general wrapper method
	
	input:
	    ix: int, list, slice, tuple (on integer index or axis values)
	    **kwargs: see help on LocatorAxis

	return:
	    `int`, list of `int` or slice of `int`
	
	"""
	#if method is None: method = self.method
	if len(kwargs) > 0:
	    self = self.set(**kwargs)

	if self.keepdims and not self._islist(ix) and not type(ix) is slice:
	    ix = [ix]

	return self[ix]

    def set(self, **kwargs):
	""" convenience function for chained call: update methods and return itself 
	"""
	#self.method = method
	dict_ = self.__dict__.copy()
	dict_.update(kwargs)
	return self.__class__(**dict_)

    #
    # locate single values
    #
    def locate(self, val):
	""" locate with try/except checks
	"""
	if not self._check_type(val):
	    raise TypeError("{}: locate: wrong type {} --> {}".format(self.__class__, type(val), val))

	try:
	    res = self._locate(val)

	except IndexError, msg:
	    if self.raise_error:
		raise
	    else:
		res = None

	return res

    def _check_type(self, val): 
	return True

    def _locate(self, val):
	""" locate without try/except check
	"""
	raise NotImplementedError("to be subclassed")

    #
    # Access a slice
    #
    def slice(self, slice_, include_last=True):
	""" Return a slice_ object

	slice_	    : slice or tuple 
	include_last: include last element 

	Note bound checking is automatically done via "locate" mode
	This is in contrast with slicing in numpy arrays.
	"""
	# Check type
	if type(slice_) is not slice:
	    raise TypeError("should be slice !")

	start, stop, step = slice_.start, slice_.stop, slice_.step

	if start is not None:
	    start = self.locate(start)
	    if start is None: raise ValueError("{} not found in: \n {}:\n ==> invalid slice".format(start, self.values))

	if stop is not None:
	    stop = self.locate(stop)
	    if stop is None: raise ValueError("{} not found in: \n {}:\n ==> invalid slice".format(stop, self.values))
	    
	    #at this stage stop is an integer index on the axis, 
	    # so make sure it is included in the slice if required
	    if include_last:
		stop += 1

	# leave the step unchanged: it always means subsampling
	return slice(start, stop, step)

class ObjLocator(LocatorAxis):
    """ locator axis for strings
    """
    def _locate(self, val):
	""" find a string
	"""
	try:
	    return self.values.tolist().index(val)
	except ValueError, msg:
	    raise IndexError(msg)


class NumLocator(LocatorAxis):
    """ Locator for axis of integers or floats to be treated as numbers (with tolerance parameters)

    Examples:
    ---------
    >>> values = np.arange(1950.,2000.)
    >>> values  # doctest: +ELLIPSIS
    array([ 1950., ... 1999.])
    >>> loc = NumLocator(values)   
    >>> loc(1951) 
    1
    >>> loc([1960, 1980, 1999])		# a list if also fine 
    [10, 30, 49]
    >>> loc(slice(1960,1970))		# or a tuple/slice (latest index included)
    slice(10, 21, None)
    >>> loc[1960:1970] == _		# identical, as any of the commands above
    True
    >>> loc([1960, -99, 1999], raise_error=False)  # handles missing values
    [10, None, 49]

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

    # Modulo
    >>> loc = NumLocator(np.array([0, 180, 360]), modulo=360)
    >>> loc[180] == loc[-180]
    True
    """
    def __init__(self, *args, **kwargs):

	# extract parameters specific to NumLocator
	opt = {'tol':1e-8, 'modulo':None}
	for k in kwargs.copy():
	    if k in opt:
		opt[k] = kwargs.pop(k)

	super(NumLocator, self).__init__(*args, **kwargs)

	self.__dict__.update(opt)
	#print self.indexing

    def _check_type(self, val):
	return isnumber(val)

    def _locate(self, val):
	""" 
	"""
	values = self.values

	# modulo calculation, val = val +/- modulo*n, where n is an integer
	# e.g. longitudes has modulo = 360
	if self.modulo is not None:

	    if not isnumber(self.modulo):
		raise TypeError("modulo parameter need to be a number, got {} --> {}".format(type(self.modulo), self.modulo))
			
	    #mi, ma = values.min(), values.max() # min, max
	    mi, ma = self.min(), self.max() # min, max

	    if self.modulo and (val < mi or val > ma):
		val = _adjust_modulo(val, self.modulo, mi)

	# locate value in axis
	loc = np.argmin(np.abs(val-values))

	if np.abs(values[loc]-val) > self.tol:
	    raise IndexError("%f not found within tol %f (closest match %i:%f)" % (val, self.tol, loc, values[loc]))

	return loc

    def min(self):
	return self.values.min()
    def max(self):
	return self.values.max()

def isnumber(val):
    try:
	val+1
	if val == 1: pass # only scalar allowed
	return True

    except:
	return type(val) != bool

class RegularAxisLoc(NumLocator):
    """ Locator for numerical axis with monotonically increasing, regularly spaced values
    """
    def min(self):
	return self.values[0]

    def max(self):
	return self.values[-1]

    @property
    def step(self):
	return self.values[1] - self.values[0]

    
def _adjust_modulo(val, modulo, min=0):
    oldval = val
    mval = np.mod(val, modulo)
    mmin = np.mod(min, modulo)
    if mval < mmin:
	mval += modulo
    val = min + (mval - mmin)
    assert np.mod(val-oldval, modulo) == 0, "pb modulo"
    return val

#    mode: different modes to handle out-of-mode situations
#	"raise": raise error
#	"clip" : returns 0 or -1 (first or last element)
#	"wrap" : equivalent to modulo=values.ptp()
#    tol: tolerance to find data 
#    modulo: val = val +/- modulo*n, where n is an integer (default None)
#
#    output:
#    -------
#    loc: integer position of val on values
#
#    Examples:
#    ---------
#
#    >>> values = [-4.,-2.,0.,2.,4.]
#    >>> locate_num(values, 2.)
#    3
#    >>> locate_num(values, 6, modulo=8)
#    1
#    >>> locate_num(values, 6, mode="wrap")
#    1
#    >>> locate_num(values, 6, mode="clip")
#    -1
#    """
#if mode != "raise":
#    if regular is None:
#	regular = is_regular(values)
#    if not regular:
#	warnings.warning("%s mode only valid for regular axes" % (mode))
#	mode = "raise"
#
#if mode == "raise":
#    raise OutBoundError("%f out of bounds ! (min: %f, max: %f)" % (val, mi, ma))
#
#elif mode == "clip":
#    if val < mi: return 0
#    else: return -1
#
#elif mode == "wrap":
#    span = values[-1] - values[0]
#    val = _adjust_modulo(val, modulo=span, min=mi)
#    assert val >= mi and val <= ma, "pb wrap"
#
#else:
#    raise ValueError("invalid parameter: mode="+repr(mode))

def make_multiindex(ix, n):
    # Just add slice(None) if some indices are missing
    ix = np.index_exp[ix] # make it a tuple

    for i in range(n-len(ix)):
	ix += slice(None),

    return ix

#
# Return a slice for an axis
#
class LocatorAxes(object):
    """ return indices over multiple axes
    """
    def __init__(self, axes, **opt):
	"""
	"""
	assert isinstance(axes, list), "must be list of axes objects"
	#assert isinstance(axes, list) and (len(axes)>0 or isinstance(axes[0], Axis)), "must be list of axes objects"
	self.axes = axes
	self.opt = opt

    def set(self, **kwargs):
	""" convenience function for chained call: update methods and return itself 
	"""
	return LocatorAxes(self.axes, **kwargs)

    def __getitem__(self, indices):
	"""
	"""
	# Construct the indices
	indices = make_multiindex(indices, len(self.axes))  # make it the right size

	numpy_indices = ()
	for i, ix in enumerate(indices):
	    loc = self.axes[i].loc(ix, **self.opt)
	#    assert np.isscalar(loc) \
	#	    or type(loc) is slice \
	#	    or type(loc) in (np.ndarray, list) and np.asarray(loc).dtype != np.dtype('O'), \
	#	    "pb with LocatorAxis {} => {}".format(ix,loc)
	    numpy_indices += loc,

	return numpy_indices

    def __call__(self, indices, axis=0, **opt):
	"""
	"""
	# convert to nd tuple
	if type(indices) is tuple or isinstance(indices, dict):
	    assert axis in (None, 0), "cannot have axis > 0 for tuple (multi-dimensional) indexing"

	if type(indices) is not tuple:
	    
	    # format (indices=..., axis=...)
	    if not isinstance(indices, dict):
		kw = {self.axes[axis].name:indices}

	    # dictionary
	    else:
		assert axis in (None, 0), "cannot have axis > 0 for tuple (multi-dimensional) indexing"
		kw = indices

	    # make sure the fields match
	    for k in kw:
		dims = [ax.name for ax in self.axes]
		if k not in dims:
		    raise ValueError("invalid axis name, present:{}, got:{}".format(dims, k))

	    # dict: just convert to appropriately ordered tuple
	    indices = ()
	    for ax in self.axes:
		if ax.name in kw:
		    ix = kw[ax.name]
		else:
		    ix = slice(None)
		indices += ix,

	kwargs = self.opt.copy()
	kwargs.update(opt)
	return LocatorAxes(self.axes, **kwargs)[indices]


def test():
    """ test module
    """
    import doctest
    import axes
    #reload(axes)
    #globs = {'Locator':Locator}
    #doctest.debug_src(Locator.__doc__)
    doctest.testmod(axes, optionflags=doctest.ELLIPSIS)


if __name__ == "__main__":
    test()


