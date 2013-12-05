""" laxarray : Labelled Axis Array

Added Behaviour to NanArray: 
-----------------------------

           + add a name to each dimension 
	   + can give an axis name to `axis=`
	   + multidimensional slicing
	   + recursive `endomorphic` transformation on subsets of the whole space

Full Genealogy:
---------------
basearray => nanarray => laxarray => dimarray

basearray: just emulate a numpy array
           + introduce the xs method (only one axis)
           + axis=0 by default (except in squeeze)
	   + `==` operator returns a boolean and not an array

nanarray : treats nans as missing values

laxarray : + add a name to each dimension 
	   + can give an axis name to `axis=`
	   + multidimensional slicing
	   + recursive `endomorphic` transformation on subsets of the whole space

dimarray : add values and metadata to each axis and to the array
	   + netCDF I/O
"""

import numpy as np
import copy

import nanarray as na
import basearray as ba
from tools import _operation

class LaxArray(na.NanArray):
    """ Contains a numpy array with named axes, and treat nans as missing values

    a = LaxArray(values, ("lat","lon"))

    Most usual numpy axis-transforms are implemented in LaxArray 
    The new feature is that the `axis` parameter can now be given an axis name instead
    of just an integer index !

    a.mean(axis='lon') :: a.mean(axis=1)
    """

    #
    #  INIT
    #

    def __init__(self, values, names=None, dtype=None, copy=False):
	""" instantiate a ndarray with values: note values must be numpy array

	values, dtype, copy: passed to np.array 
	names : axis names
	"""
	# make sure values is a numpy array
	values = np.array(values, dtype=dtype, copy=copy)

	# give axes default names if not provided
	if names is None:
	    names = ["x{}".format(i) for i in range(values.ndim)]

	for nm in names:
	    if type(nm) is not str:
		print "found:", nm
		raise ValueError("axis names must be strings !")

	#names = list(names)
	names = tuple(names)
	if len(names) != values.ndim:
	    raise ValueError("Axis names do not match array dimension: {} for ndim = {}".format(names, values.ndim))

	self.values = values
	self.names = names

    @classmethod
    def __constructor(cls, values, names, dtype=None, copy=False):
	""" just used for subclassing
	"""
	# try returning subclass
	if cls._subok:
	    try:
		return cls(values, names)
	    except:
		print "returns LaxArray !"
		pass

	return LaxArray(values=values, names=names, dtype=dtype, copy=copy)

    def copy(self):
	""" 
	>>> b = a.copy()
	>>> b == a
	True
	>>> b[0] += 1
	>>> b == a
	False
	"""
	return self.__constructor(self.values.copy(), names=copy.copy(self.names))

    def __repr__(self):
	""" screen printing

	Example:
	-------
	>>> la.array(np.zeros((5,6)),["lat","lon"])
	dimensions(5, 6): lat, lon
	array([[ 0.,  0.,  0.,  0.,  0.,  0.],
	       [ 0.,  0.,  0.,  0.,  0.,  0.],
	       [ 0.,  0.,  0.,  0.,  0.,  0.],
	       [ 0.,  0.,  0.,  0.,  0.,  0.],
	       [ 0.,  0.,  0.,  0.,  0.,  0.]])
	"""
	dims = ", ".join([nm for nm in self.names]).strip()
	header = "dimensions{}: {}".format(self.shape, dims)
	return "\n".join([header, repr(self.values)])

    #
    #  MISC
    #

    #
    # BASIC OVERLOADING AND OPERATIONS
    #

    def __eq__(self, other): 
	""" Test equality between two array, as a whole (not element by element)

	>>> b = a.copy()
	>>> a == b
	True
	>>> a == a+1
	False
	>>> b.names = ('x','y')
	>>> a == b
	False
	"""
	return isinstance(other, self.__class__) and np.all(self.values == other.values) and self.names == other.names

    def __float__(self):  return float(self.values)
    def __int__(self):  return int(self.values)

    #
    # AXIS ALIGMENT
    #

    def transpose(self, axes=None):
	""" Analogous to numpy, but also allow axis names
	>>> a          # doctest: +ELLIPSIS
	dimensions(5, 6): lat, lon
	array(...)
	>>> a.T         # doctest: +ELLIPSIS
	dimensions(6, 5): lon, lat
	array(...)
	>>> a.transpose([1,0]) == a.T == a.transpose(['lon','lat'])
	True
	"""
	if axes is not None:
	    axes = [self._get_axis_idx(ax) for ax in axes]
	    names = [self.names[i] for i in axes]

	else:
	    names = [self.names[i] for i in 1,0] # work only 2D

	result = self.values.transpose(axes)
	return self.__constructor(result, names=names)

    def squeeze(self, axis=None):
	""" Analogous to numpy, but also allows axis name

	>>> b = a.take([0], axis='lat')
	>>> b
	dimensions(1, 6): lat, lon
	array([[ 0.,  1.,  2.,  3.,  4.,  5.]])
	>>> b.squeeze()
	dimensions(6,): lon
	array([ 0.,  1.,  2.,  3.,  4.,  5.])
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

	return self.__constructor(res, names=names)


    def conform_to(self, dims):
	""" make the array conform to a list of dimension names by transposing
	dimensions and adding new ones if necessary

	dims: str, list or tuple of names

	>>> a              # doctest: +ELLIPSIS
	dimensions(5, 6): lat, lon
	array(...)
	>>> a.conform_to(('lat','lon','time'))  # doctest: +ELLIPSIS
	dimensions(5, 6, 1): lat, lon, time
	array(...)
	>>> a.conform_to(('lon','time','lat'))  # doctest: +ELLIPSIS
	dimensions(6, 1, 5): lon, time, lat
	array(...)
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

	    obj = self.__constructor(val, names=newnames)

	# and transpose to the desired dimension
	obj = obj.transpose(dims)
	assert obj.names == dims, "problem in transpose: {} != {}".format(obj.names, dims,)
	return obj


    def _operation(self, func, other):
	""" make an operation: this include axis alignment

	>>> ts = la.array(np.arange(10), ('time',))
	>>> ts*a          # doctest: +ELLIPSIS
	dimensions(10, 5, 6): time, lat, lon
	array(...)

	But beware, it is not commutative ! 

	>>> a*ts  # doctest: +ELLIPSIS
	dimensions(5, 6, 10): lat, lon, time
	array(...)

	The convention chosen here is, when in doubt, to adopt the dimension ordering
	of the left-hand side operand.
	But it is easy to permute the dimensions to any desired order:

	>>> (a*ts).transpose(('time','lat','lon'))  # doctest: +ELLIPSIS
	dimensions(10, 5, 6): time, lat, lon
	array(...)
	"""
	values, names = _operation(func, self, other)
	return self.__constructor(values, names=names)


    #
    # SLICING
    #

    def __getitem__(self, item):
	""" slice array and return the new object with correct labels

	note with pure slicing (slice object) there is no error bound check !

	>>> a[:,0:5:2]  # check against equivalent xs example
	dimensions(5, 3): lat, lon
	array([[ 0.,  2.,  4.],
	       [ 1.,  3.,  5.],
	       [ 2.,  4.,  6.],
	       [ 3.,  5.,  7.],
	       [ 4.,  6.,  8.]])
	"""
	# standard numpy
	newvalues = self.values[item]

	#
	# determine the new names
	#

	# make sure we have a tuple
	item = np.index_exp[item]

	# loop over the slices and remove a dimension if integer

	n = len(item) # first n dimensions susceptible to be squeezed
	newnames = []
	for i, nm in enumerate(self.names):

	    # squeeze out this dimension
	    if i < n and isinstance(item[i], int):
		continue

	    newnames.append(nm)

	obj = self.__constructor(newvalues, names=newnames)

	# OR USING xs:

	#args = [(slice_, self.names[i]) for i,slice_ in enumerate(item)]
	#obj = self
	#for slice_, axis in args:
	#    obj = obj.xs(slice_, axis)

	return obj

    def take(self, indices, axis=None, out=None):
	""" analogous to numpy
	"""
	# flattened array makes no sense regarding dimensions, return numpy
	if axis is None:
	    return self.values.take(indices, axis=axis, out=out)

	axis = self._get_axis_idx(axis) 

	# convert to integer axis
	res = self.values.take(indices, axis=axis, out=out)
	#names = self._get_reduced_names(axis)
	return self.__constructor(res, names=self.names)


    #
    # AXIS TRANSFORM
    #

    def _get_axis_idx(self, axis):
	""" always return an integer axis
	"""
	if type(axis) is str:
	    axis = self.names.index(axis)

	return axis

    def _get_reduced_names(self, axis):
	""" return the new axis names after an axis reduction
	"""
	axis = self._get_axis_idx(axis) # just in case
	return [nm for nm in self.names if nm != self.names[axis]]


    def apply(self, funcname, axis=None, skipna=True):
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

	>>> a.mean(axis='lon') == a.mean(axis=1)
	True

	>>> a.mean(axis=0)
	dimensions(6,): lon
	array([ 2.,  3.,  4.,  5.,  6.,  7.])

	>>> a.mean()
	4.5
	"""
	if axis is not None:
	    axis = self._get_axis_idx(axis) 

	res = super(LaxArray, self).apply(funcname, axis=axis, skipna=skipna)

	# return if scalar
	if not isinstance(res, ba.BaseArray):
	    return res

	if funcname.startswith("cum"):
	    names = self.names
	else:
	    names = self._get_reduced_names(axis) # name after reduction

	return self.__constructor(res, names=names)


    #
    # EXPERIMENTAL: recursively apply a transformation which takes a laxarray as argument 
    # and returns a laxarray: apply low dimensional functions and build up
    #

    def apply_recursive(self, axes, trans, *args, **kwargs):
	""" recursively apply a multi-axes transform

	input:
	    axes   : tuple of axis names representing the dimensions accepted by fun
			Recursive call until the `set` of dimensions match (no order required)

	    trans     : transformation from Dimarray to Dimarray to (recursively) apply

	    *args, **kwargs: arguments passed to trans in addition to the axes

	IDEA BEHIND THAT: 2-D interpolation on a n-D array... (TO TEST)

	Now a 2D function applied on a 4D array, for the sake of the example but with litte demonstrative value 
	(TO DO in dimarray with acutall interpolation)

	>>> d = la.array(np.arange(2), ['time']) 
	>>> e = la.array(np.arange(2), ['sample']) 
	>>> a4D = d * a * e # some 4D data over time, lat, lon, sample
	>>> a4D  # doctest: +ELLIPSIS
	dimensions(2, 5, 6, 2): time, lat, lon, sample
	array(...)

	Let's consider the 2D => 2D identity function on lat, lon
	(it returns an error if called on something else)

	def f(x):
	    assert set(x.names) == set(("lon","lat")), "error"
	    return x #  some 2D transform, return itself

	>>> a4D.apply_recursive(('lon','lat'), f)  # doctest: +ELLIPSIS
	dimensions(2, 2, 5, 6): time, sample, lat, lon
	array(...)

	Note how "factorized" dimensions are now at the front
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
	new = self.__constructor(data, names=newnames)

	return new
#

def array(*args, **kwargs):
    """ creates a LaxArray, convenience function
    """
    return LaxArray(*args, **kwargs)

array.__doc__ += LaxArray.__init__.__doc__

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
    a = LaxArray(values, ("lat","lon"))
    return a

def _load_test_glob():
    """ return globs parameter for doctest.testmod
    """
    # also go into locals()
    import doctest
    import numpy as np
    from laxarray import LaxArray
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

def test_readme(**kwargs):
    import doctest, sys
    return doctest.testfile("README", optionflags=doctest.ELLIPSIS,globs=_load_test_glob(),**kwargs)


if __name__ == "__main__":
    debug_doc()
    debug_readme()
