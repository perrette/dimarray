""" This module emulates a numpy array with named axes
"""
import numpy as np
import copy

# 
# For debugging
#
def get_testdata():
    # basic dataset used for debugging
    lat = np.arange(5)
    lon = np.arange(6)
    lon2, lat2 = np.meshgrid(lon, lat)
    values = np.array(lon2+lat2, dtype=float) # use floats for the sake of the example
    a = Laxarray(values, ("lat","lon"))
    return a

def debug(raise_on_error=False, **kwargs):
    import doctest, sys, ipdb
    import laxarray as la

    a = get_testdata()

    # just get this module
    m = sys.modules[__name__]

    if 'globs' not in kwargs:
	kwargs['globs'] = globals()
	kwargs['globs'].update(locals())

    return doctest.testmod(m, raise_on_error=raise_on_error, **kwargs)

def _take_slice(values, names, slice_nd):
    """ take a slice in a numpy array with labels

    input:
	values: array-like
	names : axis labels
	slice_nd: tuple of slices or integers, or single slice or integer

    returns:
	sliced_values : values[slice_nd]
	sliced_names  : corresponding names
    
    aim: make sure new object with correct labels
    """
    if type(slice_nd) is not tuple:
	slice_nd = (slice_nd, )

    for slice_ in slice_nd:
	if type(slice_) not in [int, slice]:
	    raise ValueError("`slice_nd` must be a tuple of slices or integers")

    msg = "`values` argument must be array-like"
    assert hasattr(values, "ndim"), msg
    assert hasattr(values, "shape"), msg

    sliced_values = values[slice_nd] # not affected by the above processing

    # get the names
    sliced_names = []
    for i in range(values.ndim):
	axis = np.arange(values.shape[i])

	if i >= len(slice_nd): 
	    newaxis = axis[:]
	else:
	    newaxis = axis[slice_nd[i]]

	# if collapsed axis, drop the name
	if np.iterable(newaxis):
	    sliced_names.append(names[i])

    return sliced_values, sliced_names

class Laxarray(object):
    """ Contains a numpy array with named axes, and treat nans as missing values

    a = laxarray(values, ("lat","lon"))

    Most usual numpy axis-transforms are implemented in Laxarray 
    The new feature is that the `axis` parameter can now be given an axis name instead
    of just an integer index !

    a.mean(axis='lon') :: a.mean(axis=1)

    The transforms return a Laxarray, except when `axis` is None, in which case 
    they will return a numpy array. 

    As a difference to numpy, the default value for an axis is 0 instead 
    of None, as it is natural in many cases and more consistent for named axes, 
    in that sense that a flattened array (numpy's axis=None) mixes up dimensions
    which is opposite to the spirit of a Laxarray where every labelled axis
    may have a physical meaning.

    If a numpy like behaviour is desired (with None as default), it is always 
    possible to call the underlying numpy object.

    a.mean(axis=None) :: a.values.mean()

    Examples:
    --------
    Let's make some example data
    >>> lat = np.arange(5)
    >>> lon = np.arange(6)
    >>> lon2, lat2 = np.meshgrid(lon, lat)
    >>> values = np.array(lon2+lat2, dtype=float) # use floats for the sake of the example

    Here without explict naming of the axes, just as np.array 
    (note laxarray is an alias for the Laxarray class)

    >>> a = laxarray(values)
    >>> a
    dimensions(5, 6): x0, x1
    array([[ 0.,  1.,  2.,  3.,  4.,  5.],
	   [ 1.,  2.,  3.,  4.,  5.,  6.],
	   [ 2.,  3.,  4.,  5.,  6.,  7.],
	   [ 3.,  4.,  5.,  6.,  7.,  8.],
	   [ 4.,  5.,  6.,  7.,  8.,  9.]])

    Or more meaningfull:
    >>> a = laxarray(values, ("lat","lon"))
    >>> a
    dimensions(5, 6): lat, lon
    array([[ 0.,  1.,  2.,  3.,  4.,  5.],
	   [ 1.,  2.,  3.,  4.,  5.,  6.],
	   [ 2.,  3.,  4.,  5.,  6.,  7.],
	   [ 3.,  4.,  5.,  6.,  7.,  8.],
	   [ 4.,  5.,  6.,  7.,  8.,  9.]])

    Like a numpy array:
    >>> a.shape
    (5, 6)
    >>> a.ndim
    2
    >>> a.size
    30

    >>> b = a + 2
    >>> b = a - a
    >>> b = a / 2
    >>> b = a ** 2
    >>> b = a * np.array(a)

    >>> b
    dimensions(5, 6): lat, lon
    array([[  0.,   1.,   4.,   9.,  16.,  25.],
           [  1.,   4.,   9.,  16.,  25.,  36.],
           [  4.,   9.,  16.,  25.,  36.,  49.],
           [  9.,  16.,  25.,  36.,  49.,  64.],
           [ 16.,  25.,  36.,  49.,  64.,  81.]])

    Including axis alignment:
    >>> c = laxarray(lon, ['lon'])
    >>> c
    dimensions(6,): lon
    array([0, 1, 2, 3, 4, 5])

    >>> c*b
    dimensions(6, 5): lon, lat
    array([[   0.,    0.,    0.,    0.,    0.],
           [   1.,    4.,    9.,   16.,   25.],
           [   8.,   18.,   32.,   50.,   72.],
           [  27.,   48.,   75.,  108.,  147.],
           [  64.,  100.,  144.,  196.,  256.],
           [ 125.,  180.,  245.,  320.,  405.]])

    Even with new dimensions
    >>> d = laxarray(np.arange(2), ['time']) 
    """ 
    _debug = False  # for debugging (rather use denbug function above)

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
    def _laxarray_api(cls, values, names, dtype=None, copy=False):
	""" internal api for laxarray constructor, can be overloaded for subclassing
	"""
	return cls(values=values, names=names, dtype=dtype, copy=copy)

    def __repr__(self):
	""" screen printing

	Example:
	-------
	>>> laxarray(np.zeros((5,6)),["lat","lon"])
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

    def __getitem__(self, item):
	""" slice array and return the new object with correct labels

	note with pure slicing (slice object) there is no error bound check !

	>>> a[:,-200:5:2]  # check against equivalent xs example
	dimensions(5, 3): lat, lon
	array([[ 0.,  2.,  4.],
	       [ 1.,  3.,  5.],
	       [ 2.,  4.,  6.],
	       [ 3.,  5.,  7.],
	       [ 4.,  6.,  8.]])
	"""
	# make sure we have a tuple
	item = np.index_exp[item]

	args = [(slice_, self.names[i]) for i,slice_ in enumerate(item)]

	obj = self
	for slice_, axis in args:
	    obj = obj.xs(slice_, axis)

	### Testing ###
	if self._debug:
	    val = self.values[item]
	    assert np.all(obj.values == val), "error while slicing !"

	return obj


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

    def xs(self, slice_=None, axis=0, keepdims=False, **kwargs):
	""" Multi-dimensional slicing

	First mode:

	slice_  : integer, list, slice or tuple
	axis    : axis to slice
	keepdims: [False] keep initial dimensions
	**kwargs: keyword arguemnts with axis names as keys "<axis> = <slice_>"

	Various behaviours depending on the type slice_
	int  :  slice along axis, squeeze it out
	list :  sample several int slices, same shape as before
	tuple:  ([start,] stop[, step]), similar to slice, except that the 
		stop element *is* included and a bound check is performed
	slice:  (experimental): similar to a[start:stop:step] 
		where start, stop and step are slice attributes
		Note that in numpy, bounds are not checked when performing slicing, beware !

	Examples:
	--------
	>>> a
	dimensions(5, 6): lat, lon
	array([[ 0.,  1.,  2.,  3.,  4.,  5.],
	       [ 1.,  2.,  3.,  4.,  5.,  6.],
	       [ 2.,  3.,  4.,  5.,  6.,  7.],
	       [ 3.,  4.,  5.,  6.,  7.,  8.],
	       [ 4.,  5.,  6.,  7.,  8.,  9.]])

	Equivalent expressions to get the 3rd line:

	>>> a.xs(3, axis=0)   # integer axis
	dimensions(6,): lon
	array([ 3.,  4.,  5.,  6.,  7.,  8.])

	>>> a.xs(3, axis='lat')  # labelled axis
	dimensions(6,): lon
	array([ 3.,  4.,  5.,  6.,  7.,  8.])

	>>> a.xs(lat=3, lon=5)	      # multiple keyword arguments
	dimensions(): 
	array(8.0)

	Take several columns

	>>> a.xs(lon=[0,3,4])		# as a list
	dimensions(5, 3): lat, lon
	array([[ 0.,  3.,  4.],
	       [ 1.,  4.,  5.],
	       [ 2.,  5.,  6.],
	       [ 3.,  6.,  7.],
	       [ 4.,  7.,  8.]])

	>>> a.xs((3,5), axis='lon')	# as a tuple 3 to 5 (included)
	dimensions(5, 3): lat, lon
	array([[ 3.,  4.,  5.],
	       [ 4.,  5.,  6.],
	       [ 5.,  6.,  7.],
	       [ 6.,  7.,  8.],
	       [ 7.,  8.,  9.]])

	>>> a.xs(lon=(3,5,2))		# as a tuple step > 1
	dimensions(5, 2): lat, lon
	array([[ 3.,  5.],
	       [ 4.,  6.],
	       [ 5.,  7.],
	       [ 6.,  8.],
	       [ 7.,  9.]])

	>>> a.xs(slice(-200,5,2), axis='lon') # as a slice (last index missing, no bound checking)
	dimensions(5, 3): lat, lon
	array([[ 0.,  2.,  4.],
	       [ 1.,  3.,  5.],
	       [ 2.,  4.,  6.],
	       [ 3.,  5.,  7.],
	       [ 4.,  6.,  8.]])
	"""
	# recursive definition

	# keyword arguments
	if slice_ is None:

	    if len(kwargs) == 0:
		newobj = self

	    else:
		axis, slice_ = kwargs.popitem()
		obj = self.xs(slice_, axis=axis, keepdims=keepdims)
		newobj = obj.xs(keepdims=keepdims, **kwargs)

	    return newobj

	#
	# assume len(kwargs) is zero
	#
	assert len(kwargs) == 0, "cannot give both explicit and keyword arguments"

	axis = self._get_axis_idx(axis) 
	n = self.shape[axis]

	# convert tuple to slice with appropriate changes and checks
	# bound checking + including last element
	if type(slice_) is tuple:
	    slice_ = slice(*slice_)

	    if slice_.stop is not None:
		assert slice_.stop < n, "beyond max axis bounds !"
		slice_ = slice(slice_.start, slice_.stop+1, slice_.step) # include last element !

	    if slice_.start is not None:
		assert slice_.start >= 0, "outside axes bounds !"

	# basic definition
	if type(slice_) is int:
	    indices = [slice_]
	    squeeze = True

	elif type(slice_) is slice:
	    idx = np.arange(n)
	    indices = idx[slice_]
	    squeeze = False

	elif type(slice_) is list:
	    indices = slice_
	    squeeze = False

	else:
	    print slice_
	    print type(slice_)
	    raise ValueError("`slice_` may be int, tuple or list only")

	res = self.take(indices, axis=axis)

	# if collapsed axis, drop the name
	if not keepdims and squeeze:
	    res = res.squeeze(axis)

	return res

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

	return self._laxarray_api(res, names)

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
	return self._laxarray_api(res.copy(), names=copy.copy(self.names))

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
	    res = res.filled()
	return res

    def mean(self, axis=0, skipna=True):
	""" Return the mean along an axis

	input:
	    axis   : int or str (the axis label), or None [default 0] 
	    skipna : if True, skip the NaNs in the computation

	return:
	    laxarray: meaned Laxarray (if axis is not None, else np.ndarray)

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

	# return a float if axis is None
	if axis is None:
	    return values.mean()

	axis = self._get_axis_idx(axis) 
	res_ma = values.mean(axis=axis)
	res = self._array(res_ma) # fill nans back in if necessary

	names = self._get_reduced_names(axis) # name after reduction
	return self._laxarray_api(res, names=names)

    def median(self, axis=0, skipna=True):
	""" Return the median along an axis

	see `mean` documentation for more information
	"""
	values = self._values(skipna)

	# return a float if axis is None
	if axis is None:
	    return values.mean()

	axis = self._get_axis_idx(axis) 
	res_ma = values.median(axis=axis)
	res = self._array(res_ma) # fill nans back in if necessary
	names = self._get_reduced_names(axis) # name after reduction
	return self._laxarray_api(res, names=names)

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
	    laxarray (if axis is not None, else numpy array)
	"""
	if axis is None:
	    return self.values.take(indices, out=out)

	# convert to integer axis
	axis = self._get_axis_idx(axis) 
	res = self.values.take(indices, axis=axis, out=out)
	#names = self._get_reduced_names(axis)
	return self._laxarray_api(res, names=self.names)

    def clip(self, a_min, a_max, out=None):
	""" analogous to numpy's `clip` method
	"""
	res = self.values.clip(a_min=a_min, a_max=a_max, out=out)
	return self._laxarray_api(res, names=self.names)

    def compress(self, condition, axis=0, out=None):
	""" logical indexing along an axis (analogous to numpy)
	"""
	if axis is None: 
	    return self.values.compress(condition, out=out)

	axis = self._get_axis_idx(axis)
	res = self.values.compress(condition, axis=axis, out=out) # does not reduce
	return self._laxarray_api(res, names=self.names)

    def permute(self, axes, inplace=False):
	"""  permute an array and its axes (like numpy's transpose, see transpose method which handles the 2D case)

	input:
	    axes: new axes order, can be a list of str names or of integers
	
	note: axes must include all dimensions, but can include more of them w/o error raising
	"""
	axes = [self._get_axis_idx(ax) for ax in axes]

	# re-arrange array
	newvalues = self.values.transpose(axes)

	# re-arrange names
	newnames = tuple([self.names[i] for i in axes])

	if inplace:
	    self.values = newvalues
	    self.names = names

	else:
	    return self._laxarray_api(newvalues, newnames)

    def transpose(self, axes=None, **kwargs):
	""" to be consistent with numpy 
	""" 
	if axes is None:
	    assert self.ndim == 2, "transpose only 2D unless axes is provided"
	    axes = 1,0
	return self.permute(axes, **kwargs)

    def conform_to(self, dims):
	""" make the array conform to a list of dimension names

	dims: str, list or tuple of names
	"""
	dims = tuple(dims)
	assert type(dims[0]) is str, "input dimensions must be strings"

	if self.names == dims:
	    return self

	# check that all dimensions are present, otherwise add
	if set(self.names) != set(dims):
	    missing_dims = tuple([nm for nm in dims if not nm in self.names])
	    newnames = missing_dims + self.names

	    val = self.values.copy()
	    for nm in missing_dims:
		val = val[np.newaxis] # prepend singleton dimension

	    obj = self._laxarray_api(val, newnames)

	else:
	    obj = self

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

    def __add__(self, other): return _operation(np.add, self, other, laxarray=self._laxarray_api)

    def __mul__(self, other): return _operation(np.multiply, self, other, laxarray=self._laxarray_api)

    def __sub__(self, other): return _operation(np.subtract, self, other, laxarray=self._laxarray_api)

    def __div__(self, other): return _operation(np.divide, self, other, laxarray=self._laxarray_api)

    def __pow__(self, other): return _operation(np.power, self, other, laxarray=self._laxarray_api)

    def __eq__(self, other): 
	return isinstance(other, Laxarray) and np.all(self.values == other.values) and self.names == other.names


laxarray = Laxarray # as convenience function


#
# Handle operations 
#
def _unique(nm):
    """ return the same ordered list without duplicated elements
    """
    new = []
    for k in nm:
	if k not in new:
	    new.append(k)
    return new

def _operation(func, o1, o2, align=True, order=None, laxarray=laxarray):
    """ operation on Laxarray objects

    input:
	func	: operator
	o1, o2	: operand
	align, optional: if True, use pandas to align the axes
	order	: if order is True, align the dimensions along a particular order

	laxarray: constructor (takes values and axes)

    output:
	Laxarray
    """
    # second operand is not a Laxarray: let numpy do the job 
    if not isinstance(o2, Laxarray): 
	if np.ndim(o2) > o1.ndim:
	    raise ValueError("bad input: second operand's dimensions not documented")
	res = func(o1.values, np.array(o2))
	return laxarray(res, o1.names)

    # if same dimension: easy
    if o1.names == o2.names:
	res = func(o1.values, o2.values)
	return laxarray(res, o1.names)

    # otherwise determine the dimensions of the result
    if order is None:
	order = _unique(o1.names + o2.names) 
    else:
	order = [o for o in order if o in o1.names or o in o2.names]
    order = tuple(order)

    #
    o1 = o1.conform_to(order)
    o2 = o2.conform_to(order)

    assert o1.names == o2.names, "problem in transpose"
    res = func(o1.values, o2.values) # just to the job

    return laxarray(res, order)

# # For later
# 
# #    Many usual numpy axis-transforms are aliased on a Laxarray. The additional
# #    new feature is that the `axis` parameter can now be given an axis name instead
# #    of just an integer index !
# #
# #    >>> a.mean(axis='lon')
# #    dimensions(5,): lat
# #
# #    But the integer index also work
# #
# #	>>> la.mean(axis=1)
# #	dimensions(5,),: lat
# #
# #    Cumulative transformation also works:
# #
# #	>>> la.cumsum(axis='lat')
# #	dimensions(5,6): lat x lon
# #
# #    As well as taking percentiles (note 'lat' becomes 'pct_lat'):
# #
# #	>>> la.percentile([5,50,95],axis='lat')
# #	dimensions(3,6): pct_lat x lon
# #
# #    There is additionally a new xs method which takes slice along a given axis
# #
# #	>>> a.xs(4, axis="lon")
# #	>>> a.xs(lon=, lat=45.5)
# #	dimarray: 30 non-null elements (0 null)
# #	name: "", units: "", descr: "", lat: "45.5"
# #	dimensions: lon
# #	lon (30): 30.5 to 59.5
# #
# #    Note the tuple here indicates a slice, whereas a list just samples individual
# #    data points, quite consistently with what you would expect from standard slicing.
# #
# #	>>> a.xs(lon=[30.5, 60.5], lat=45.5)
# #	dimarray: array([ 0.76372818,  2.15288386])
# #	name: "", units: "", descr: "", lat: "45.5"
# #	dimensions: lon
# #	lon (2): 30.5 to 60.5
# #
# #    Let's check the consistency just in case:
# #
# #	>>> a.xs(lon=(30.5, 60.5), lat=45.5).mean()
# #	dimarray: array(-0.00802083197740012)
# #
# #	>>> a[45.5, 30.5:60.5].mean()
# #	dimarray: array(-0.00802083197740012)
