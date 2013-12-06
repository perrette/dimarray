import numpy as np
from collections import OrderedDict
import string
import copy

__all__ = ["Axis","Axes"]

#
# Methods and class to slice dimensions
#


def _get_slicer_method(value):
    """ automatically determine the slice method 
    """
    # set default method based on the axis type

    # exact for string and integer
    if type(value) in [str, int]:
	method = "index"

    # nearest for float
    else:
	method = "nearest"
    return method

#
# Axis class
#
class Axis(object):
    """ Axis

    name  : name (attribute)
    values: numpy array (or list) 

    also has methods to help indexing
    """
    def __init__(self, values, name="", units="", descr="", slicing=None, **attrs):
	""" 
	"""
	if not name:
	    assert hasattr(values, "name"), "unnamed dimension !"
	    name = values.name # e.g pandas axis

	## collapsed axis
	#if not np.iterable(values)
	#    values = [values]
	#else:
	values = _fix_type(values)

	self.values = values

	if slicing is None:
	    slicing = _get_slicer_method(values[0])
	self.slicing = slicing # default method to access a dimension
	self.attrs = attrs

    def ncattrs(self):
	""" return attributes to be written to netCDF file
	"""
	return [nm for nm in self.__dict__ if nm not in ["values","slicing"]]

    def __getitem__(self, item):
	""" access values elements & return an axis object
	"""
	#return Axis(self.values[item], self.name) 
	return self.values[item]

    def __getattr__(self, att):
	""" Access attributes
	"""
	# Locator attributes (index etc...)
	#if hasattr(Locator, att) and not att.startswith("_"):
	    #return getattr(self.loc, att)

	# Or values attributes (e.g. len, size...)
	if hasattr(self.values, att):
	    return getattr(self.values, att)

	else:
	    return getattr(self.loc, att)

    def __eq__(self, other):
	return isinstance(other, Axis) and np.all(other.values == self.values)

    @property
    def loc(self):
	""" Access the slicer to locate axis elements
	"""
	return Locator(self.values, self.slicing)

    def __repr__(self):
	""" string representation
	""" 
	return "{} ({}): {} to {}".format(self.name, self.size, self.values[0], self.values[-1])

    def issingleton(self):
	return np.size(self.values) == 1

    def copy(self):
	return copy.copy(self)

def default_axes_names(n):
    """ make up default axis names following "x0", "x1", ...
    """
    return ["x{}".format(i) for i in range(n)]

def default_axis_name(i):
    return "x{}".format(i)


class Axes(list):
    """ Axes class: inheritates from a list but dict-like access methods
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
		    name = default_axis_name(len(self))

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
	body = "\n".join([repr(ax) for ax in self])
	return "\n".join([header, body])

    def sort(self, names):
	""" sort IN PLACE
	"""
	if type(names[0]) is int:
	    names = self[names].names # names to sort along
	super(Axes, self).sort(key=lambda x: names.index(x.name))

    def copy(self):
	return copy.copy(self)

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
# Return a slice for an axis
#

class Locator(object):
    """ Return a slice for an Axis

    Examples:
    >>> loc = Locator(axis.values, method='index')
    >>> loc[35.2:60.1] # by default
    >>> loc.set(method='index')[35.2:60.1] # same as above but make sure method is "index"
    >>> loc.set(method='nearest')[35.:60.] # index of nearest value
    >>> loc.set(method='nearest')[35.:60.] # index of nearest value
    """
    def __init__(self, values, method=None):
	"""
	values	: string list or numpy array
	method	: default method: "index" or "nearest" (or numpy)
	"""
	self.values = values
	self.method = method

    def __getitem__(self, ix):
	""" return a slice

	>>> loc[34.3]
	>>> loc[34.3:67.]
	>>> loc[34.3:67:5]
	"""
	# if not a slice: easy
	if type(ix) is not slice:
	    return self(ix)

	# else, update start/stop/step
	start, stop, step = ix.start, ix.stop, ix.step

	if start is not None:
	    start = self(start)

	if stop is not None:
	    stop = self(stop)

	if step is not None:
	    step = int(round(step/self.step)) # corresponding step on the index

	return slice(start, stop, step)

    @property
    def step(self):
	return self.values[1] - self.values[0]

    @property
    def size(self):
	return np.size(self.values)

    def __call__(self, idx, method = None, slice_=False):
	""" locate an index

	>>> loc(34.3)
	>>> loc(34.3, method='nearest') # can choose the method
	>>> loc([34.3, 56])		# locate two values
	>>> loc([34.3, 56], slice_=True)# return a slice between the two values

	Note: the slice_ argument can be useful for scripting, e.g. when 
	      a reference period can be provided as a single year or a tuple
	"""
	if method is None:
	    method = self.method

	loc = getattr(self, method) 

	if type(idx) is not str and np.iterable(idx):
	    # here must distinguish two cases:
	    # ... (v1, v2) ==> slice (e.g. to pass xs(time=(1999,2010)) )
	    # ... [v1, v2, ...] ==> iterate
	    if not type(idx) is tuple: 
		slice_ = False 
	    idx = [loc(ix) for ix in idx]

	    # make it a slice if required
	    if slice_:	    #and len(idx) == 2:
		idx = slice(*idx)

	    return idx

	else:
	    return loc(idx)

    #
    # methods to access single value
    #
    def index(self, val):
	""" locate a slice

	val: index value
	"""
	val = type(self.values[0])(val) # convert to the same type first
	return list(self.values).index(val)

    exact = index # alias for index, by opposition to nearest

    def nearest(self, val, check_bounds = True):
	""" return nearest index
	"""
	if check_bounds:
	    dx = self.step
	    mi, ma = np.min(self.values), np.max(self.values)
	    if val < mi - dx/2 or val > ma + dx/2:
		raise ValueError("%f out of bounds ! (min: %f, max: %f, step: %f)" % (val, mi, ma, dx))
	return np.argmin(np.abs(val-self.values))

    def numpy(self, idx):
	""" numpy index
	"""
	return idx



#
# Return a slice for N axes
#
class LocatorND(object):
    """ Find a slice for a n-D array given its axes, similar to numpy's index_exp

    There is a new copy of this class for each access to axes.loc, with a different method parameter
    """
    def __init__(self, axes):
	""" axes: list of Axis
	"""
	self.axes = axes
	self.method = None

    def __getitem__(self, item):
	""" in case the dimension is known
	"""
	item = np.index_exp[item]
	slice_ = ()

	for i, ax in enumerate(self.axes):
	    if i >= len(item): 
		slice_ += np.index_exp[:]
		continue

	    ax = self.axes[i]
	    #ix = ax.loc[item[i]]
	    if self.method is not None:
		method = self.method
	    else:
		method = ax.slicing

	    ix = Locator(ax.values, method)[item[i]]
	    slice_ += np.index_exp[ix]

	return slice_

    def __call__(self, method=None, slice_=True, **kwargs):
	""" return a n-D slice 
	"""
	ix = ()

	# loop over the axes and check the indices
	for ax in self.axes:

	    # axis not sliced: continue
	    if ax.name not in kwargs: 
		ix += np.index_exp[:]
		continue

	    if method is None: 
		method = ax.method # allow axis-dependent method

	    idx = kwargs.pop(ax.name)
	    ii = ax.loc(idx, slice_=slice_, method=method) 
	    ix += np.index_exp[ii] # append the tuple of indices

	if len(kwargs) > 0:
	    raise ValueError("invalid axes: "+kwargs.keys())

	return ix


