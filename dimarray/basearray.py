""" basearray : Base Array Container

Behaviour: 
----------

    - same as numpy array, except that
	- axis=0 by default (except in squeeze)
	- == returns a boolean and not an array
    - add the xs method along one axis only

Full Genealogy:
---------------

BaseArray => NanArray => LaxArray => DimArray

BaseArray: just emulate a numpy array
           + introduce the xs method (only one axis)
	   + `==` operator returns a boolean and not an array
	   recent update: axis is None by default like in numpy

NanArray : treats nans as missing values

LaxArray : add a name to each dimension 
	   + can give an axis name to `axis=`
	   + multidimensional slicing
	   + recursive `endomorphic` transformation on subsets of the whole space

DimArray : add values and metadata to each axis and to the array
	   + netCDF I/O

"""
import numpy as np
import copy
from tools import _slice_to_indices, _convert_dtype, _operation, _check_scalar, _NumpyDesc

class BaseArray(object):
    """ Contains a numpy array with named axes, and treat nans as missing values

    As a difference to numpy, the default behavious for an along-axis transform
    if to take `axis=0` instead of None. 
    
    This behaviour is experimental and may change is the future to be more 
    consistent with numpy arrays.

    The transforms return a BaseArray, except when `axis` is None, in which case 
    they will return a numpy array. 

    If a numpy like behaviour is desired (with None as default), it is always 
    possible to call the underlying numpy object.

    a.mean(axis=None) :: a.values.mean()

    Additionally, a new `xs` method is introduced, similar to `take`
    """

    # class-attributes to determine how a slice should work
    _check_bounds = False
    _include_last = False

    _subok = True # if True, will return a subclass when possible, else, will return a basearray

    def __init__(self, values, dtype=None, copy=False):
	""" instantiate a ndarray with values: note values must be numpy array

	values: first argument passed to np.array()
	dtype, copy: passed to np.array 
	"""
	self.values = np.array(values, dtype=dtype, copy=copy)

    @classmethod
    def _constructor(cls, values):
	""" just used for subclassing
	"""
	# try returning subclass
	if cls._subok:
	    try:
		return cls(values)
	    except:
		print "returns BaseArray !"
		pass

	return BaseArray(values)

    def __getitem__(self, item):
	""" slice array and return the new object with correct labels
	"""
	val = self.values[item]
	return self._constructor(values=val)

    def __setitem__(self, item, val):
	""" set item: numpy method
	"""
	self.values[item] = val

    def __repr__(self):
	""" screen printing
	"""
	header = "<BaseArray>"
	return "\n".join([header,repr(self.values)])

    def squeeze(self, axis=None):
	""" remove singleton dimensions
	"""
	res = self.values.squeeze(axis=axis)
	return self._constructor(values=res)

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
	return self._constructor(self.values.copy())

    def apply(self, funcname, axis=None):
	""" apply along-axis numpy method
	"""
	assert type(funcname) is str, "can only provide function as a string"

	# call numpy method
	values = getattr(self.values, funcname)(axis=axis) 

	if isinstance(values, np.ndarray):
	    return self._constructor(values)

	# just return the scalar
	else:
	    return values

    def take(self, indices, axis=0):
	""" apply along-axis numpy method
	"""
	res = self.values.take(indices, axis=axis)
	return self._constructor(res)

    def transpose(self, axes=None):
	""" apply along-axis numpy method
	"""
	res = self.values.take(indices, axes)
	return self._constructor(res)

    #
    # Add numpy transforms
    #
    mean = _NumpyDesc("apply", "mean")
    median = _NumpyDesc("apply", "median")
    sum  = _NumpyDesc("apply", "sum")
    diff = _NumpyDesc("apply", "diff")
    prod = _NumpyDesc("apply", "prod")
    min = _NumpyDesc("apply", "min")
    max = _NumpyDesc("apply", "max")
    ptp = _NumpyDesc("apply", "ptp")
    cumsum = _NumpyDesc("apply", "cumsum")
    cumprod = _NumpyDesc("apply", "cumprod")

    @property
    def T(self):
	return self.transpose()

    #
    # basic operattions
    #
    def _operation(self, func, other):
	""" make an operation

	Just for testing:
	>>> b
	<BaseArray>
	array([[ 0.,  1.],
	       [ 1.,  2.]])
	>>> b == b
	True
	>>> b + 2
	<BaseArray>
	array([[ 2.,  3.],
	       [ 3.,  4.]])
	>>> b+b == b*2
	True
	>>> b*b == b**2
	True
	>>> (b - b.values) == b - b
	True
	"""
	if isinstance(other, BaseArray):
	    res = func(self.values, other.values)
	else:
	    res = func(self.values, other)

	return self._constructor(res)

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
    # ADD-ON TO NUMPY: AN XS METHOD (along-axis only for basearray)
    #

    def _xs_axis(self, idx, axis=0, _keepdims=False, _scalar_conversion=True, _include_last=None, _check_bounds=None):
	""" Take a slice/cross-section along one axis

	input:
	    idx     : integer, list, slice or tuple
	    axis    : axis to slice
	    _keepdims: [False], for integer slicing only: keep initial dimensions

	    _scalar_conversion [True], if True, return a scalar whenever size == 1

	output:
	    instance of BaseArray or scalar

	Various behaviours based on idx type:

	- int  :  slice along axis, squeeze it out, may return a scalar 
	- list :  same as `take`: sample several int slices, same shape as before =
	- tuple:  alias for slicing, but performing on individual axes as well

		slice_ = slice(*tuple), where slice_ has parameters start, stop, step 
		(see python built-in slice documentation)
		a.xs((start, stop, step), axis=0) :: a[start:stop:step]
		
		Note that similarly to numpy, bounds are not checked when performing slicing, beware !
		TO DO: class-wide parameters _check_bounds and _include_last 

	Examples:
	--------
	>>> a
	<BaseArray>
	array([[ 0.,  1.,  2.,  3.,  4.,  5.],
	       [ 1.,  2.,  3.,  4.,  5.,  6.],
	       [ 2.,  3.,  4.,  5.,  6.,  7.],
	       [ 3.,  4.,  5.,  6.,  7.,  8.],
	       [ 4.,  5.,  6.,  7.,  8.,  9.]])

	Equivalent expressions to get the 3rd line:

	>>> a.xs(3, axis=0)   
	<BaseArray>
	array([ 3.,  4.,  5.,  6.,  7.,  8.])

	Take several columns

	>>> a.xs([0,3,4], axis=1)  # as a list
	<BaseArray>
	array([[ 0.,  3.,  4.],
	       [ 1.,  4.,  5.],
	       [ 2.,  5.,  6.],
	       [ 3.,  6.,  7.],
	       [ 4.,  7.,  8.]])

	>>> a.xs((3,5), axis=1)	  # as a tuple
	<BaseArray>
	array([[ 3.,  4.],
	       [ 4.,  5.],
	       [ 5.,  6.],
	       [ 6.,  7.],
	       [ 7.,  8.]])

	>>> a.xs((3,5), axis=1) == a[:,3:5]  # similar to
	True

	>>> a.xs((0,5,2), axis=1)	# as a tuple step > 1
	<BaseArray>
        array([[ 0.,  2.,  4.],
               [ 1.,  3.,  5.],
               [ 2.,  4.,  6.],
               [ 3.,  5.,  7.],
               [ 4.,  6.,  8.]])

	"""
	if type(idx) is list:
	    res = self.take(idx, axis=axis)

	elif type(idx) is int:
	    res = self.take([idx], axis=axis)

	    # if collapsed axis, drop the name
	    if not _keepdims:
		res = res.squeeze(axis=axis)

	elif type(idx) is tuple:
	    res = self._take_tuple(idx, axis=axis, check_bounds=_check_bounds, include_last=_include_last)

	else:
	    print idx
	    print type(idx)
	    raise ValueError("`idx` may be int, tuple or list only")

	# If result is scalar, like, convert to scalar unless otherwise specified 
	if _scalar_conversion:
	    res, test = _check_scalar(res)

	return res

    #
    # for a base array, no way to do multidimensional slicing
    #
    xs = _xs_axis

    def _take_tuple(self, t, axis=0, check_bounds=None, include_last=None):
	""" similar to take, but for a `slice` instead of indices

	input:
	    t: tuple ==> slice_ = slice(*t)
	       (see python built-in slice documentation)

	    check_bounds = check min/max bound (TO DO !)
	    include_last = include last element (TO DO!)
	"""
	assert type(t) is tuple, "only accept tuple !"
	slice_ = slice(*t)

	# use default class values if not provided
	if check_bounds is None:
	    check_bounds = self._check_bounds
	if include_last is None:
	    include_last = self._include_last

	if check_bounds:
	    raise NotImplementedError("bound checking not yet implemented !")

	indices = _slice_to_indices(slice_, self.shape[axis], include_last=include_last)
	return self.take(indices, axis=axis)


array = BaseArray # as convenience function

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
    a = BaseArray(values)
    return a

def _load_test_glob():
    """ return globs parameter for doctest.testmod
    """
    # also go into locals()
    import numpy as np
    import basearray as ba

    a = get_testdata()
    values = a.values
    b = a[:2,:2] # small version

    return locals()

def test_doc(raise_on_error=False, optionflags=1, globs={}, **kwargs):
    import doctest
    import basearray as ba
    doctest.ELLIPSIS = True  # understands ...

    kwargs['globs'] = _load_test_glob()

    return doctest.testmod(ba, raise_on_error=raise_on_error, optionflags=optionflags,**kwargs)


if __name__ == "__main__":
    test_doc()
