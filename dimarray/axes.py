import numpy as np
from collections import OrderedDict
import string
import copy

from metadata import Variable, Metadata

__all__ = ["Axis","Axes"]

def _fix_type(axis):
    """ fix type for axes coming from pandas
    """
    dtype = type(axis[0])
    if dtype is str:
	axis = list(axis)
    else:
	axis = np.array(axis, dtype=dtype)

    return axis

#
# Axis class
#
class Axis(Variable):
    """ Axis

    Required Attributes:

	values: numpy array (or list) 
	name  : name (attribute)

    + metadata (see Variable class)
	=> name, units, descr, stamp are pre-defined
    """
    def __init__(self, values, name="", slicing=None, **attrs):
	""" 
	"""
	if not name:
	    assert hasattr(values, "name"), "unnamed dimension !"
	    name = values.name # e.g pandas axis

	values = _fix_type(values)

	# Initialize self._metadata and set self._values
	super(Axis, self).__init__(values, name=name)

	# Note: this is equivalent to : 
	#   self._values = values
	#   self._metadata = self._metadata_default.copy() # an (possibly empty) ordered dictionary
	#   self.name = name

    def __getitem__(self, item):
	""" access values elements & return an axis object
	"""
	return self.values[item]

    def __eq__(self, other):
	return isinstance(other, Axis) and np.all(other.values == self.values)

    @property
    def loc(self):
	""" Access the slicer to locate axis elements
	"""
	return Locator(self.values)

    @property
    def iloc(self):
	""" Access the slicer to locate axis elements
	"""
	return Locator(self.values, method="numpy")

    def __repr__(self):
	""" string representation
	""" 
	values = "{} ({}): {} to {}".format(self.name, self.size, self.values[0], self.values[-1])
	meta = self.repr_meta()
	return "\n".join(values, meta)

    def issingleton(self):
	return np.size(self.values) == 1

    def copy(self):
	return copy.copy(self)

#
# List of axes
#

class Axes(list):
    """ Axes class: inheritates from a list but dict-like access methods for convenience
    """
    def __init__(self, axes=[]):
	"""
	"""
	super(Axes, self).__init__(axes)
	for ax in self:
	    assert isinstance(ax, Axis), "use a list of Axis objects to initialize Axes"

    def append(self, axis, name=None):
	""" add a check on axis
	"""
	if not isinstance(axis, Axis):

	    if name is None:

		# name contained in the attribute
		if hasattr(axis, "name"):
		    name = axis.name

		# or guess follow default: "x0", "x1"
		else:
		    name = "x{}".format(len(self))

	    axis = Axis(axis, name)

	super(Axes, self).append(axis)

    @classmethod
    def from_tuples(cls, *tuples_name_values):
	""" initialize axes from tuples

	>>> Axes.from_tuples(('lat',mylat), ('lon',mylon)) 
	"""
	newaxes = Axes()
	for nm, values in tuples_name_values:
	    newaxes.append(values, nm)
	return newaxes

    @classmethod
    def from_list(cls, values, names=None):
	""" 
	"""
	if names is None: 
	    names = [None for __ in range(len(values))]

	return cls.from_tuples(*zip(names, values))

    @classmethod
    def from_dict(cls, names=None, shape=None, **kwargs):
	""" infer dimensions from key-word arguments
	"""
	axes = cls()
	for k in kwargs:
	    axes.append(kwargs[k], k)

	# Make sure the order is right (since it is lost via dict-passing)

	# preferred solution: names is given
	if names is not None:
	    axes.sort(names)

	# alternative option: only the shape is given
	elif shape is not None:
	    current_shape = [ax.size for ax in axes]
	    assert len(set(current_shape)) == len(set(axes.names)), \
    """ some axes have the same size !
    ==> ambiguous determination of dimensions order via keyword arguments only
    ==> explictly supply `names=` or use from_list() or from_tuples() methods" """
	    argsort = [shape.index(k) for k in current_shape]
	    axes = axes[argsort]

	    current_shape = tuple([ax.size for ax in axes])
	    assert current_shape == shape, "dimensions mismatch !"

	return axes

    def __getitem__(self, item):
	""" get an axis by integer or name
	"""
	if type(item) is str:
	    item = self.names.index(item)

	if np.iterable(item):
	    return Axes([self[i] for i in item])

	return super(Axes,self).__getitem__(item)
	#return super(Axes,self)[item]

    def __repr__(self):
	""" string representation
	"""
	header = "dimensions: "+ " x ".join(self.names)
	body = "\n".join([repr(ax).split('\n')[0] for ax in self])
	return "\n".join([header, body])

    def sort(self, names):
	""" sort IN PLACE
	"""
	if type(names[0]) is int:
	    names = self[names].names # names to sort along
	super(Axes, self).sort(key=lambda x: names.index(x.name))

    def copy(self):
	return copy.copy(self)


#
# Return a slice for an axis
#

class Locator(object):
    """ This class is the core of indexing in dimarray. 

	loc = Locator(values [, method])  

    where `values` represent the axis values
    and `method` is one of:

    "numpy"  : integer-based indexing 
    "exact"  : exact matching of the value to locate in `values`
    "nearest": nearest match (with bound checking)

    By default `method` is assumed to be "exact" for axes made of `int`
    and `str`, and "nearest" for `float`.

    A locator instance is generated from within the Axis object, via 
    its properties loc (valued-based indexing) and iloc (integer-based)

	(a) axis.loc  ==> Locator(values)  

	(b) axis.iloc ==> Locator(values , method="numpy")  

    A locator is hashable is a similar way to a numpy array, but also 
    callable to provide the "method" parameter on-the-fly.

    It returns an integer index or `list` of `int` or `slice` of `int` which 
    is understood by numpy's arrays. In particular we have:

	loc[ix] = np.index_exp[loc[ix]][0]

    The "set" method can also be useful for chained calls. We have the general 
    equivalence:

	loc(idx, method) :: loc.set(method)[idx]

    except when idx is a tuple, in which case it becomes

	loc(idx, method) :: loc.set(method)[slice(*idx)]


    Examples:
    ---------
    >>> values = np.arange(1950.,2000.)
    >>> values  # doctest: +ELLIPSIS
    array([1950., ... ,1999.])
    >>> loc = Locator(values)   
    >>> loc(1951) 
    1
    >>> loc(1951.4) # also works in nearest approx
    1
    >>> loc(1951.4, method="exact")     # unless "exact" or "index" is specified
    Exception ...
    >>> loc([1960, 1980, 2000])		# a list if also fine 
    [10, 30, 50]
    >>> loc((1960,1970))		# or a tuple/slice (latest index included)
    >>> loc[1960:1970] == _		# identical, as any of the commands above
    True

    Test equivalence with np.index_exp
    >>> ix = 1951
    >>> loc[ix] = np.index_exp[loc[ix]][0]
    True
    >>> ix = [1960, 1980, 2000]
    >>> loc[ix] = np.index_exp[loc[ix]][0]
    True
    >>> ix = slice(1960,1970)
    >>> loc[ix] = np.index_exp[loc[ix]][0]
    True
    >>> ix = 1951
    >>> loc[ix] = np.index_exp[loc[ix]][0]
    True
    """
    def __init__(self, values, method=None):
	"""
	values	: string list or numpy array

	method	: default method: "index", "nearest", or "numpy"
	"""
	self.values = values

	if method is None:
	    method = self._method

	self.method = method #  default method

    def _method(self):
	""" default slicing method
	"""
	dtype = type(self.values[0])

	# exact for integer or string
	if dtype in [str, int]: 
	    method = "index"

	# nearest for float
	else: 
	    method = "nearest"
	return method

    #
    # wrapper methods: __getitem__ and __call__
    #

    def __getitem__(self, ix):
	""" best at handling slices

	>>> loc[34.3]
	>>> loc[34.3:67.]
	>>> loc[34.3:67:5]
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
	val = type(self.values[0])(val) # convert to the same type first
	return list(self.values).index(val)

    exact = index # alias for index, by opposition to nearest

    def nearest(self, val, check_bounds = True):
	""" return nearest index with bound checking
	"""
	if check_bounds:
	    dx = self.step
	    mi, ma = np.min(self.values), np.max(self.values)
	    if val < mi - dx/2 or val > ma + dx/2:
		raise ValueError("%f out of bounds ! (min: %f, max: %f, step: %f)" % (val, mi, ma, dx))
	return np.argmin(np.abs(val-self.values))

    def numpy(self, idx):
	""" numpy index: do nothing
	"""
	return idx

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

    def slice(self, slice_, method=None):
	""" Return a slice_ object

	slice_: slice or tuple 
	"""
	# Check type
	assert type(slice_) in (tuple, slice), "must provide a slice or a tuple"
	if type(slice_) is tuple:
	    slice_ = slice(*slice_)

	# update method
	if method is None: method = self.method

	# include last element, except for numpy-like indexing
	# (for consistency with numpy and pandas)
	# not bound checking is automatically done in both "index" 
	# ("exact") and "nearest" methods, but not in "numpy"
	# again, this is consistent with other packages.
	if method  == "numpy":
	    include_last = False
	else:
	    include_last = True

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

def test():
    """ test module
    """
    import doctest
    doctest.debug_src(Locator.__doc__)

if __name__ == "__main__":
    #test()
    import doctest
    doctest.debug_src(Locator.__doc__)

