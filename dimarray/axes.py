import numpy as np
from collections import OrderedDict
import string
import copy

from metadata import Metadata

__all__ = ["Axis","Axes"]


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
class Axis(Metadata):
    """ Axis

    Required Attributes:

	values: numpy array (or list) 
	name  : name (attribute)

    """
    _metadata_exclude = ("values", "name", "weights") # variables which are NOT metadata

    def __init__(self, values, name="", weights=None, slicing=None, **attrs):
	""" 
	"""
	if not name:
	    assert hasattr(values, "name"), "unnamed dimension !"
	    name = values.name # e.g pandas axis

	# if string, make it an object-typed array
	if isinstance(type(values[0]), str) or type(values[0]) is np.str_:
	    values = np.array(values, dtype=object) # make sure data is stored with object type

	# otherwise, just a normal array
	else:
	    values = np.array(values, copy=False)

	super(Axis, self).__init__() # init an ordered dict of metadata

	self.values = values 
	self.name = name 
	self.weights = weights 

    def get_weights(self):
	""" return axis weights as a Dimarray
	"""
	from core import Dimarray

	# no weights
	if self.weights is None:
	    weights = np.ones_like(self.values)

	# function of axis
	elif callable(self.weights):
	    weights = self.weights(self.values)

	# same weight for every element
	elif np.size(self.weights) == 1:
	    weights = np.zeros_like(self.values) + self.weights

	# already an array of weights
	else:
	    weights = np.array(weights, copy=False)

	# index on one dimension
	ax = Axis(self.values, name=self.name)

	return Dimarray(weights, ax)

    def __getitem__(self, item):
	""" access values elements & return an axis object
	"""
	values = self.values[item]

	# if collapsed to scalar, just return it
	if not isinstance(values, np.ndarray):
	    return values

	if isinstance(self.weights, np.ndarray):
	    weights = self.weights[item]

	else:
	    weights = self.weights

	return Axis(values, self.name, weights=weights)

    def __eq__(self, other):
	return isinstance(other, Axis) and np.all(other.values == self.values)

    @property
    def loc(self):
	""" Access the slicer to locate axis elements
	"""
	return Locator(self.values)

    def __repr__(self):
	""" string representation
	""" 
	values = "{} ({}): {} to {}".format(self.name, self.size, self.values[0], self.values[-1])
	
	meta = self.repr_meta()
	return "\n".join([values, meta])

    def issingleton(self):
	return np.size(self.values) == 1

    def copy(self):
	return copy.copy(self)

    # a few array-like properties
    @property
    def size(self): 
	#return np.size(self.values)
	return self.values.shape[0]

    @property
    def dtype(self): 
	return _convert_dtype(np.array(self.values).dtype)


#
# List of axes
#

class Axes(list):
    """ Axes class: inheritates from a list but dict-like access methods for convenience
    """
    def append(self, item):
	""" add a check on axis
	"""
	# if item is an Axis, just append it
	assert isinstance(item, Axis), "can only append an Axis object !"
	super(Axes, self).append(item)

    @classmethod
    def from_tuples(cls, *tuples_name_values):
	""" initialize axes from tuples

	Axes.from_tuples(('lat',mylat), ('lon',mylon)) 
	"""
	assert type(tuples_name_values[0]) is tuple, "need to provide a list of `name, values` tuples !"

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
    def from_list(cls, list_of_arrays, dims=None):
	"""  list of np.ndarrays and dims
	"""
	assert type(list_of_arrays) in (list, tuple), "need to provide a list of arrays !"

	# default names
	if dims is None: 
	    dims = ["x{}".format(i) for i in range(len(list_of_arrays))]

	return cls.from_tuples(*zip(dims, list_of_arrays))

    @classmethod
    def from_kwds(cls, dims=None, shape=None, **kwargs):
	""" infer dimensions from key-word arguments
	"""
	# if no key-word argument is given, just return default axis
	if len(kwargs) == 0:
	    return cls.from_shape(shape, dims)

	axes = cls()
	for k in kwargs:
	    axes.append(Axis(kwargs[k], k))

	# Make sure the order is right (since it is lost via dict-passing)

	# preferred solution: dims is given
	if dims is not None:
	    axes.sort(dims)

	# alternative option: only the shape is given
	elif shape is not None:
	    current_shape = [ax.size for ax in axes]
	    assert len(set(current_shape)) == len(set([ax.name for ax in axes])), \
    """ some axes have the same size !
    ==> ambiguous determination of dimensions order via keyword arguments only
    ==> explictly supply `dims=` or use from_list() or from_tuples() methods" """
	    argsort = [current_shape.index(k) for k in shape]
	    axes = axes[argsort]

	    current_shape = tuple([ax.size for ax in axes])
	    assert current_shape == shape, "dimensions mismatch (axes shape: {} != values shape: {}".format(current_shape, shape)

	else:
	    raise Warning("no shape information: random order")

	return axes

    def __getitem__(self, item):
	""" get an axis by integer or name
	"""
	if type(item) is str:
	    item = self.get_idx(item)

	if np.iterable(item):
	    return Axes([self[i] for i in item])

	return super(Axes,self).__getitem__(item)
	#return super(Axes,self)[item]

    def __repr__(self):
	""" string representation
	"""
	header = "dimensions: "+ " x ".join([repr(ax.name) for ax in self])
	body = "\n".join(["{} / {}".format(i, repr(ax).split('\n')[0]) for i,ax in enumerate(self)])
	return "\n".join([header, body])

    def sort(self, dims):
	""" sort IN PLACE according to the order in "dims"
	"""
	if type(dims[0]) is int:
	    dims = [ax.name for ax in self]

	super(Axes, self).sort(key=lambda x: dims.index(x.name))

    def copy(self):
	return copy.copy(self)

    def get_idx(self, axis):
	""" always return axis integer location
	"""
	# if axis is already an integer, just return it
	if type(axis) is int:
	    return axis

	dims = [ax.name for ax in self]

	return dims.index(axis)

#
# Return a slice for an axis
#

class Locator(object):
    """ This class is the core of indexing in dimarray. 

	loc = Locator(values [, method])  

    where `values` represent the axis values
    and `method` is one of:

    "exact"  : exact matching of the value to locate in `values`
    "nearest": nearest match (with bound checking)

    By default `method` is assumed to be "exact" for axes made of `int`
    and `str`, and "nearest" for `float`.

    A locator instance is generated from within the Axis object, via 
    its properties loc (valued-based indexing) and iloc (integer-based)

	axis.loc  ==> Locator(values)  

    A locator is hashable is a similar way to a numpy array, but also 
    callable to provide the "method" parameter on-the-fly.

    It returns an integer index or `list` of `int` or `slice` of `int` which 
    is understood by numpy's arrays. In particular we have:

	loc[ix] == np.index_exp[loc[ix]][0]

    The "set" method can also be useful for chained calls. We have the general 
    equivalence:

	loc(idx, method) :: loc.set(method)[idx]

    except when idx is a tuple, in which case it becomes

	loc(idx, method) :: loc.set(method)[slice(*idx)]


    Examples:
    ---------
    >>> values = np.arange(1950.,2000.)
    >>> values  # doctest: +ELLIPSIS
    array([ 1950., ... 1999.])
    >>> loc = Locator(values)   
    >>> loc(1951) 
    1
    >>> loc(1951.4) # also works in nearest approx
    1
    >>> loc(1951.4, method="exact")     # doctest: +ELLIPSIS
    Exception raised:
	...
	ValueError: 1951.4000000000001 is not in list
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
    def __init__(self, values, method=None):
	"""
	values	: string list or numpy array

	method	: "index" (same as "exact") or "nearest" for float

	NOTE: for str or int, "index" is always used
	"""
	self.values = values

	# default method: "nearest" (for float)
	if method is None:
	    method = "nearest"

	# but set to "index" if str or int valued axis
	dtype = type(values[0])
	if dtype in [str, int]: 
	    method = "index"

	self.method = method #  default method

    #
    # wrapper methods: __getitem__ and __call__
    #

    def __getitem__(self, ix):
	""" 
	"""
	if type(ix) is slice:
	    return self.slice(ix)

	elif type(ix) is list:
	    return self.take(ix)

	# int, float, string
	else:
	    return self.locate(ix)

    def __call__(self, idx, method = None):
	""" general wrapper method
	
	input:
	    idx: int, list, slice, tuple (on integer index or axis values)

	    method: see class documentation

	return:
	    `int`, list of `int` or slice of `int`
	
	"""
	if type(idx) is tuple:
	    idx = slice(*idx)

	if method is None: method = self.method
	return self.set(method=method)[idx]

    def set(self, method):
	""" convenience function for chained call: update methods and return itself 
	"""
	self.method = method
	return self

    #
    # methods to access single value
    #
    def index(self, val):
	""" exactly locate a value on the axis (use built-in list `index` method)
	val: index value
	"""
	#val = type(self.values[0])(val) # convert to the same type first
	return list(self.values).index(val)

    exact = index # alias for index, by opposition to nearest

    def nearest(self, val, check_bounds = True):
	""" return nearest index with bound checking
	"""
	if check_bounds:
	    dx = self.step
	    mi, ma = np.min(self.values), np.max(self.values)
	    if val < mi - dx/2. or val > ma + dx/2.:
		raise ValueError("%f out of bounds ! (min: %f, max: %f, step: %f)" % (val, mi, ma, dx))
	return np.argmin(np.abs(val-self.values))

    #
    # wrapper for single value
    #

    def locate(self, val, method=None):
	""" wrapper to locate single values
	"""
	if method is None: method = self.method
	return getattr(self, method)(val)

    #
    # Access a list
    #

    def take(self, indices, method=None):
	""" Return a list of indices
	"""
	assert type(indices) is list, "must provide a list !"
	if method is None: method = self.method
	return map(getattr(self, method), indices)

    #
    # Access a slice
    #

    def slice(self, slice_, method=None, include_last=True):
	""" Return a slice_ object

	slice_	    : slice or tuple 
	include_last: include last element 

	Note bound checking is automatically done via "locate" method
	This is in contrast with slicing in numpy arrays.
	"""
	# Check type
	assert type(slice_) in (tuple, slice), "must provide a slice or a tuple"
	if type(slice_) is tuple:
	    slice_ = slice(*slice_)

	# update method
	if method is None: method = self.method

	start, stop, step = slice_.start, slice_.stop, slice_.step

	if start is not None:
	    start = self.locate(start, method=method)

	if stop is not None:
	    stop = self.locate(stop, method=method)
	    
	    #at this stage stop is an integer index on the axis, 
	    # so make sure it is included in the slice if required
	    if include_last:
		stop += 1

	# leave the step unchanged: it always means subsampling
	return slice(start, stop, step)

    @property
    def step(self):
	return self.values[1] - self.values[0]

    @property
    def size(self):
	return np.size(self.values)

#def _common_axes(axes1, axes2):
#    """ Align axes which have a dimension in common
#
#    input:
#	axes1, axes2: Axes objects
#    """
#    # common list of dimensions
#    dims1 =  [ax.name for ax in axes1] 
#    dims =  [ax.name for ax in axes2 if ax.name in dims1] 
#
#    newaxes = []
#    for name in dims:
#	newaxis = _common_axis(axes1[name], axes2[name])
#	newaxes.append(newaxis)
#
#    return Axes(newaxes)

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


