""" basearray : Base Array Container

Behaviour: 
----------

    - same as numpy array, except that
	- axis=0 by default (except in squeeze)
	- == returns a boolean and not an array
    - add the xs method along one axis only

Full Genealogy:
---------------

basearray => nanarray => laxarray => dimarray

basearry: just emulate a numpy array
nanarray: treats nans as missing values
laxarray: add a name to each dimension 
    can give an axis name to `axis=`
dimarray: add values and metadata to each axis and to the array
"""
import numpy as np
import copy
from functools import partial
from shared import slice_to_indices as _slice_to_indices

#
# Descriptors to add functions following same patterns
#
class _Desc(object):
    """ to apply a numpy function which reduces an axis
    """
    def __init__(self, apply, func_name):
	"""
	apply: `apply` method 
		as a function: 
		as a str     : will look for bound method in the described object, useful for subclassing
        func_name   : name of the function to apply
	"""
	self.func_name = func_name
	self.apply = apply

    def __get__(self, obj, cls=None):
	"""
	"""
	# convert self.apply to an actual function
	if self.apply is str:
	    apply = getattr(obj, self.apply) 
	else:
	    apply = self.apply

	newmethod = partial(apply, obj, self.func_name)
	newmethod.__doc__ = apply.__doc__.replace("`func`","`"+self.func_name+"`")
	return newmethod

class BaseArray(object):
    """ Contains a numpy array with named axes, and treat nans as missing values
    """

    # class-attributes to determine how a slice should work
    _check_bounds = False
    _include_last = False

    def __init__(self, values, dtype=None, copy=False):
	""" instantiate a ndarray with values: note values must be numpy array

	values: first argument passed to np.array()
	dtype, copy: passed to np.array 
	"""
	self.values = np.array(values, dtype=dtype, copy=copy)

    @classmethod
    def _constructor(cls, values, dtype=None, copy=False):
	""" constructor method, by default a BaseArray, but it can be changed 
	to something else by subclassing 
	"""
	return BaseArray(values=values, dtype=dtype, copy=copy)

    def __getitem__(self, item):
	""" slice array and return the new object with correct labels
	"""
	val = self.values[item]
	return self._constructor(val)

    def __setitem__(self, item, val):
	""" set item: numpy method
	"""
	self.values[item] = val

    def squeeze(self, axis=None):
	""" remove singleton dimensions
	"""
	res = self.values.squeeze(axis=axis)
	return self._constructor(res)

    #
    # BELOW, PURE LAYOUT FOR NUMPY TRANSFORMS
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

    def copy(self):
	return self._constructor(self.values.copy(), names=copy.copy(self.names))

    def apply(self, funcname, axis=0):
	""" apply `func` along an axis
	"""

	# retrive bound method
	method = getattr(self.values, funcname)

	# return a float if axis is None, just as numpy does
	if axis is None:
	    return method()

	values = method(axis=axis)

	return self._constructor(values)

    #
    # Add numpy transforms
    #
    mean = _Desc("apply", "mean")
    median = _Desc("apply", "median")
    sum  = _Desc("apply", "sum")
    diff = _Desc("apply", "diff")
    prod = _Desc("apply", "prod")
    min = _Desc("apply", "min")
    max = _Desc("apply", "max")
    ptp = _Desc("apply", "ptp")
    cumsum = _Desc("apply", "cumsum")
    cumprod = _Desc("apply", "cumprod")

    #
    # Lower-level numpy functions so select a subset of the array
    #

    def take(self, indices, axis=0, out=None):
	""" Analogous to numpy's `take` method, but allow axis label
	in addition to axis index.

	By contrast to numpy, default value for axis is 0 because axis names
	
	input:
	    indices: array-like
	    axis   : axis label or integer index or None
	
	output:
	    BaseArray (if axis is not None, else numpy array)
	"""
	if axis is None:
	    return self.values.take(indices, out=out)

	res = self.values.take(indices, axis=axis, out=out)
	return self._constructor(res)

    def clip(self, a_min, a_max, out=None):
	""" analogous to numpy's `clip` method
	"""
	res = self.values.clip(a_min=a_min, a_max=a_max, out=out)
	return self._constructor(res)

    def compress(self, condition, axis=0, out=None):
	""" logical indexing along an axis (analogous to numpy)
	"""
	if axis is None: 
	    return self.values.compress(condition, out=out)

	axis = self._get_axis_idx(axis)
	res = self.values.compress(condition, axis=axis, out=out) # does not reduce
	return self._constructor(res)

    def transpose(self, axes=None):
	""" analogous to numpy's transpose

	input:
	    axes: new axes order, can be a list of str names or of integers
	"""
	if axes is None:
	    # do nothing for 1-D arrays (like numpy)
	    if self.ndim == 1:
		return self

	    assert self.ndim == 2, "transpose only 2D unless axes is provided"
	    axes = 1,0

	# re-arrange array
	newvalues = self.values.transpose(axes)

	# re-arrange names
	newnames = tuple([self.names[i] for i in axes])

	if inplace:
	    self.values = newvalues
	    self.names = names

	else:
	    return self._constructor(newvalues, names=newnames)

    @property
    def T(self):
	return self.transpose()

    #
    # basic operattions
    #
    def _operation(self, func, other):
	""" make an operation
	"""
	if isinstance(other, BaseArray):
	    res = func(self.values, other.values)
	else:
	    res = func(self.values, other)

	return self._constructor(values)

    def __add__(self, other): return self._operation(np.add, other)

    def __sub__(self, other): return self._operation(np.subtract, other)

    def __mul__(self, other): return self._operation(np.multiply, other)

    def __div__(self, other): return self._operation(np.divide, other)

    def __pow__(self, other): return self._operation(np.power, other)

    def __eq__(self, other): 
	""" note it uses the np.all operator !
	"""
	return isinstance(other, self.__class__) and np.all(self.values == other.values)

    def __float__(self):  return float(self.values)
    def __int__(self):  return int(self.values)


    #
    # Here the add-on: an xs method
    #

    def _xs_axis(self, idx, axis=0, keepdims=False, _scalar_conversion=True):
	""" take a slice/cross-section along one axis

	_scalar_conversion [True], if True, return a scalar whenever size == 1
	"""
	if type(idx) is list:
	    res = self.take(idx, axis=axis)

	elif type(idx) is int:
	    res = self.take([idx], axis=axis)

	    # if collapsed axis, drop the name
	    if not keepdims:
		res = res.squeeze(axis=axis)

	elif type(idx) is tuple:
	    res = self._take_tuple(idx, axis=axis)

	elif type(idx) is slice:
	    res = self._take_slice(idx, axis=axis)

	else:
	    print idx
	    print type(idx)
	    raise ValueError("`idx` may be int, tuple or list only")

	# If result is scalar, like, convert to scalar unless otherwise specified 
	if _scalar_conversion:
	    if isinstance(self, BaseArray) and self.size == 1:
		type_ = _convert_dtype(self.dtype)
		res = type_(self.values)
	    else:
		res = self

	return res

    #
    # for a base array, no way to do multidimensional slicing
    #
    xs = _xs_axis

    def _take_slice(self, slice_, axis=0, check_bounds=None, include_last=None):
	""" similar to take, but for a `slice` instead of indices
	"""
	assert type(slice_) is slice, "only accept slice !"

	# use default class values if not provided
	if check_bounds is None:
	    check_bounds = self._check_bounds
	if include_last is None:
	    include_last = self._include_last

	if check_bounds:
	    raise NotImplementedError("bound checking not yet implemented !")

	indices = _slice_to_indices(slice_, include_last=include_last)
	return self.take(indices, axis=axis)

    def _take_tuple(self, t, axis=0, **kwargs):
	""" just to simplify syntax, shortcut for a slice

	**kwargs: passed to _take_slice
	"""
	assert type(t) is tuple, "only accept tuple !"
	slice_ = slice(*t)
	return self._take_slice(slice_, axis=axis, **kwargs)


array = BaseArray # as convenience function

if __name__ == "__main__":
    pass
