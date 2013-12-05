""" This module emulates a numpy array which treats nans as missing values, nothing else !

This is the ancestor of the family

nanarray => laxarray => dimarray
"""
import numpy as np
import copy
from functools import partial

from shared import operation as _operation

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

class NanArray(object):
    """ Contains a numpy array with named axes, and treat nans as missing values

    a = NanArray(values, ("lat","lon"))

    Most usual numpy axis-transforms are implemented in NanArray 
    The new feature is that the `axis` parameter can now be given an axis name instead
    of just an integer index !

    a.mean(axis='lon') :: a.mean(axis=1)

    The transforms return a NanArray, except when `axis` is None, in which case 
    they will return a numpy array. 

    As a difference to numpy, the default value for an axis is 0 instead 
    of None, as it is natural in many cases and more consistent for named axes, 
    in that sense that a flattened array (numpy's axis=None) mixes up dimensions
    which is opposite to the spirit of a NanArray where every labelled axis
    may have a physical meaning.

    If a numpy like behaviour is desired (with None as default), it is always 
    possible to call the underlying numpy object.

    a.mean(axis=None) :: a.values.mean()
    """

    def __init__(self, values, dtype=None, copy=False):
	""" instantiate a ndarray with values: note values must be numpy array

	values: first argument passed to np.array()
	dtype, copy: passed to np.array 
	"""
	self.values = np.array(values, dtype=dtype, copy=copy)

    @classmethod
    def _constructor(cls, values, dtype=None, copy=False):
	""" constructor method, by default a NanArray, but it can be changed 
	to something else by subclassing 
	"""
	return NanArray(values=values, dtype=dtype, copy=copy)

    def __getitem__(self, item):
	""" slice array and return the new object with correct labels
	"""
	val = self.values[item]
	return self._constructor(val)

    def __setitem__(self, item, val):
	""" set item: numpy method
	"""
	self.values[item] = val

    def xs_axis(self, idx, axis=None, keepdims=False, _scalar_conversion=True):
	""" take a slice/cross-section along one axis

	idx: 

	_scalar_conversion [True], if True, return a scalar whenever size == 1
	"""
	axis = self._get_axis_idx(axis) 

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
	    if isinstance(self, NanArray) and self.size == 1:
		type_ = _convert_dtype(self.dtype)
		res = type_(self.values)
	    else:
		res = self

	return res

    def _take_slice(self, slice_, axis=0):
	""" similar to take, but for a `slice` instead of indices
	"""
	assert type(slice_) is slice, "only accept slice !"
	n = self.shape[axis]
	idx = np.arange(n)
	indices = idx[slice_]
	return self.take(indices, axis=axis)

    def _take_tuple(self, t, axis=0):
	""" similar to _take_slice, but for a `tuple` instead of slice_
	and does bound checking
	"""
	assert type(t) is tuple, "only accept tuple !"
	axis = self._get_axis_idx(axis) 

	# convert tuple to slice with appropriate changes and checks
	# bound checking + including last element
	n = self.shape[axis]
	slice_ = slice(*t)

	if slice_.stop is not None:
	    assert slice_.stop <= n, "beyond max axis bounds, set stop to None or use slice to avoid this error message!"
	    slice_ = slice(slice_.start, slice_.stop, slice_.step) 

	if slice_.start is not None:
	    assert slice_.start >= 0, "outside axes bounds !"

	return self._take_slice(slice_, axis=axis)

    def squeeze(self, axis=None):
	""" remove singleton dimensions
	"""
	if axis is None:
	    names = [nm for i, nm in enumerate(self.names) if self.shape[i] > 1]
	    res = self.values.squeeze()

	else:
	    axis = self._get_axis_idx(axis) 
	    names = list(self.names)
	    if self.shape[axis] == 1:  # 0 is not removed by numpy...
		names.pop(axis)
	    res = self.values.squeeze(axis)

	return self._constructor(res, names=names)

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

    #
    # Operations on array that require dealing with NaNs
    #

    def _ma(self, skipna=True):
	""" return a masked array in place of NaNs if skipna is True
	"""
	a = self.values

	# return numpy array is no NaN is present of if skipna is False
	if not np.any(np.isnan(a)) or not skipna:
	    return a

	masked_array = np.ma.array(a, mask=np.isnan(a), fill_value=np.nan)
	return masked_array

    @staticmethod
    def _array(res):
	""" reverse of _ma: get back standard numpy array, 

	filling nans back in if necessary
	"""
	# fills the nans back in
	if isinstance(res, np.ma.MaskedArray): 
	    res.fill(np.na)
	return res

    def apply(self, funcname, axis=0, skipna=True):
	""" apply `func` along an axis

	Generic description:

	input:
	    funcname : string name of the function to apply (must be a numpy method)
	    axis   : int or str (the axis label), or None [default 0] 
	    skipna : if True, skip the NaNs in the computation

	return:
	    result : from the same class (if axis is not None, else np.ndarray)

	Examples:
	--------

	>>> a.mean(axis='lon')
	dimensions(5,): lat
	array([ 2.5,  3.5,  4.5,  5.5,  6.5])

	Default to axis=0 (here "lat")

	>>> a.mean()
	dimensions(6,): lon
	array([ 2.,  3.,  4.,  5.,  6.,  7.])

	>>> a.mean(axis=None)
	4.5
	"""
	values = self._ma(skipna)

	# retrive bound method
	method = getattr(values, funcname)

	# return a float if axis is None
	if axis is None:
	    return method()

	axis = self._get_axis_idx(axis) 
	res_ma = method(axis=axis)
	res = self._array(res_ma) # fill nans back in if necessary

	if funcname.startswith("cum"):
	    names = self.names
	else:
	    names = self._get_reduced_names(axis) # name after reduction

	return self._constructor(res, names=names)

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
	    NanArray (if axis is not None, else numpy array)
	"""
	if axis is None:
	    return self.values.take(indices, out=out)

	# convert to integer axis
	axis = self._get_axis_idx(axis) 
	res = self.values.take(indices, axis=axis, out=out)
	#names = self._get_reduced_names(axis)
	return self._constructor(res, names=self.names)

    def clip(self, a_min, a_max, out=None):
	""" analogous to numpy's `clip` method
	"""
	res = self.values.clip(a_min=a_min, a_max=a_max, out=out)
	return self._constructor(res, names=self.names)

    def compress(self, condition, axis=0, out=None):
	""" logical indexing along an axis (analogous to numpy)
	"""
	if axis is None: 
	    return self.values.compress(condition, out=out)

	axis = self._get_axis_idx(axis)
	res = self.values.compress(condition, axis=axis, out=out) # does not reduce
	return self._constructor(res, names=self.names)

    def transpose(self, axes=None, inplace=False):
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

	axes = [self._get_axis_idx(ax) for ax in axes]

	# re-arrange array
	newvalues = self.values.transpose(axes)

	# re-arrange names
	newnames = tuple([self.names[i] for i in axes])

	if inplace:
	    self.values = newvalues
	    self.names = names

	else:
	    return self._constructor(newvalues, names=newnames)


    def conform_to(self, dims):
	""" make the array conform to a list of dimension names by transposing
	dimensions and adding new ones if necessary

	dims: str, list or tuple of names
	"""
	dims = tuple(dims)
	assert type(dims[0]) is str, "input dimensions must be strings"
	if not set(self.names).issubset(dims):
	    raise ValueError("dimensions missing! {} cannot conform to {}. \nDid you mean {}?".format(self.names, dims, tuple(_unique(self.names+dims))))

	if self.names == dims:
	    return self

	# check that all dimensions are present, otherwise add
	if set(self.names) != set(dims):
	    missing_dims = tuple([nm for nm in dims if not nm in self.names])
	    newnames = missing_dims + self.names

	    val = self.values.copy()
	    for nm in missing_dims:
		val = val[np.newaxis] # prepend singleton dimension

	    obj = self._constructor(val, names=newnames)

	# and transpose to the desired dimension
	obj = obj.transpose(dims)
	assert obj.names == dims, "problem in transpose"
	return obj

    @property
    def T(self):
	return self.transpose()

    #
    # basic operattions
    #
    def _operation(self, func, other):
	""" make an operation
	"""
	values, names = _operation(np.add, self, other)
	return self._constructor(values, names=names)

    def __add__(self, other): return _operation(np.add, other)

    def __sub__(self, other): return _operation(np.subtract, other)

    def __mul__(self, other): return _operation(np.multiply, other)

    def __div__(self, other): return _operation(np.divide, other)

    def __pow__(self, other): return _operation(np.power, other)

    def __eq__(self, other): 
	""" note it uses the np.all operator !
	"""
	return isinstance(other, self.__class__) and np.all(self.values == other.values) and self.names == other.names

    def __float__(self):  return float(self.values)
    def __int__(self):  return int(self.values)

    #
    # Experimental: recursively apply a transformation which takes a laxarray as argument 
    # and returns a laxarray: apply low dimensional functions and build up
    #
    def apply_recursive(self, axes, trans, *args, **kwargs):
	""" recursively apply a multi-axes transform

	input:
	    axes   : tuple of axis names representing the dimensions accepted by fun
			Recursive call until the `set` of dimensions match (no order required)

	    trans     : transformation from Dimarray to Dimarray to (recursively) apply

	    *args, **kwargs: arguments passed to trans in addition to the axes

	Examples: 2-D interpolation on a 3D array... (TO TEST)

	1-D example, for testing the function
	>>> f = lambda x : x.mean('lon') # dummy function applying along "lon"
	>>> a.apply_recursive(('lon',), f)
	dimensions(5,): lat
	array([ 2.5,  3.5,  4.5,  5.5,  6.5])

	We verify that:	

	>>> a.mean('lon')
	dimensions(5,): lat
	array([ 2.5,  3.5,  4.5,  5.5,  6.5])

	Now a 2D function applied on a 4D array, for the sake of the example but with litte demonstrative value 
	(TO DO in dimarray with acutall interpolation)

	>>> d = la.array(np.arange(2), ['time']) 
	>>> e = la.array(np.arange(2), ['sample']) 
	>>> a4D = d * a * e # some 4D data over time, lat, lon, sample

	a4D has dimensions:
	dimensions(2, 5, 6, 2): time, lat, lon, sample

	Let's consider the 2D => 2D identity function on lat, lon
	(it returns an error if called on something else)

	def f(x):
	    assert set(x.names) == set(("lon","lat")), "error"
	    return x #  some 2D transform, return itself

	>>> res = a4D.apply_recursive(('lon','lat'), f)

	Note lat and lon are now last dimensions:
	dimensions(2, 2, 5, 6): time, sample, lat, lon
	"""
	#
	# Check that dimensions are fine
	# 
	assert type(axes) is tuple, "axes must be a tuple of axis names"
	assert type(axes[0]) is str, "axes must be a tuple of axis names"

	assert set(axes).issubset(set(self.names)), \
		"dimensions ({}) not found in the object ({})!".format(axes, self.names)

	#
	# If axes exactly matches: Done !
	# 
	if len(self.names) == len(axes):
	    #assert set(self.names) == set(axes), "something went wrong"
	    return trans(self, *args, **kwargs)

	#
	# otherwise recursive call
	#

	# search for an axis which is not in axes
	i = 0
	while self.names[i] in axes:
	    i+=1 

	# make sure it worked
	assert self.names[i] not in axes, "something went wrong"

	# now make a slice along this axis
	axis = self.names[i]

	# Loop over one axis
	data = []
	for k in range(self.shape[i]): # first axis

	    # take a slice of the data (exact matching)
	    slice_ = self.xs(k, axis=axis)

	    # apply the transform recursively
	    res = slice_.apply_recursive(axes, trans, *args, **kwargs)

	    data.append(res.values)

	newnames = (axis,) + res.names # new axes

	# automatically sort in the appropriate order
	new = self._constructor(data, names=newnames)

	return new


array = NanArray # as convenience function

# Update doc of descripted functions



# 
# For testing
#
# CHECK README FORMATTING python setup.py --long-description | rst2html.py > output.html

def get_testdata():
    # basic dataset used for testing
    lat = np.arange(5)
    lon = np.arange(6)
    lon2, lat2 = np.meshgrid(lon, lat)
    values = np.array(lon2+lat2, dtype=float) # use floats for the sake of the example
    a = NanArray(values, ("lat","lon"))
    return a

def _load_test_glob():
    """ return globs parameter for doctest.testmod
    """
    # also go into locals()
    import doctest
    import numpy as np
    from laxarray import NanArray
    import laxarray as la

    a = get_testdata()
    values = a.values

    # to test apply_recursive
    def f(x):
	assert set(x.names) == set(("lon","lat")), "error"
	return x #  some 2D transform, return itself

    return locals()

def test_doc(raise_on_error=False, globs={}, **kwargs):
    import doctest
    import laxarray as la

    kwargs['globs'] = _load_test_glob()

    return doctest.testmod(la, raise_on_error=raise_on_error, **kwargs)

def test_readme(optionflags=1,**kwargs):
    import doctest, sys
    doctest.ELLIPSIS = True
    return doctest.testfile("README", optionflags=optionflags,globs=_load_test_glob(),**kwargs)


if __name__ == "__main__":
    test_doc()
    test_readme()
