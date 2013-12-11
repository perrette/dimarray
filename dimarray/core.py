""" Array with dimensions
"""
import numpy as np
import copy

#import plotting

from metadata import Metadata, append_stamp
from axes import Axis, Axes
from tools import pandas_obj
import transform  # numpy axis transformation, interpolation

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

    _pandas_like = False # influences the behaviour of ix and loc

    #
    # NOW MAIN BODY OF THE CLASS
    #

    def __init__(self, values, *axes, **kwargs):
	""" Initialization

	values	: numpy-like array, or Dimarray instance
	axes	: variable list of Axis objects or list of tuples ('dim', values)
		  This argument can only be omitted if values is an instance of Dimarray

	metadata: stored in _metadata dict

	dtype, copy: passed to np.array()

	_slicing: default slicing method "numpy", "nearest", "exact" (mostly for internal use)
	"""
	# filter **kwargs and keep metadata
	default = dict(dtype=None, copy=False, _slicing=None, weights=None)
	for k in default:
	    if k in kwargs:
		default[k] = kwargs.pop(k)

	metadata = kwargs
	_slicing= default.pop('_slicing')
	weights = default.pop('weights')

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

	super(Dimarray, self).__init__() # init an ordered dict self._metadata

	self.values = avalues
	self.axes = axes

	# options
	self._slicing = _slicing
	self.weights = weights

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

    @staticmethod
    def _constructor(values, axes, **metadata):
	""" Internal API for the constructor: check whether a pre-defined class exists

	values	: numpy-like array
	axes	: Axes instance 

	This static method is used whenever a new Dimarray needs to be instantiated
	for example after a transformation.

	This makes the sub-classing process easier since only this method needs to be 
	overloaded to make the sub-class a "closed" class.
	"""
	return Dimarray(values, *axes, **metadata)

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
    def from_kwds(cls, values, dims=None, **kwaxes):
	""" initialize Dimarray with with axes as key-word arguments

	values	    : numpy-like array (passed to array())
	**kwaxes      : axes passed as keyword araguments <axis name>=<axis val>
	"""
	# "convenience", quick check on axes, to avoid a "deep" error message
	for k in kwaxes:
	    if type(kwaxes[k]) is str:
		print "PASSED:",k,":",kwaxes[k]
		msg = \
""" no attribute can be passed with the from_kwds method 
==> try using the set() method instead, for example:
a = Dimarray.from_kwds(values, **kwaxes)
a.name = "myname"
a.units = "myunits"
"""
		raise ValueError(msg)

	list_axes = Axes.from_kwds(shape=np.shape(values), dims=dims, **kwaxes)

	return cls(values, *list_axes)

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
	    super(Dimarray, self).__getattr__(att) # call Metadata's method


    def set(self, inplace=False, **kwargs):
	""" update multiple class attributes in-place or after copy

	inplace: modify attributes in-place, return None 
	otherwise first make a copy, and return new obj

	a.set(_slicing="numpy")[:30]
	a.set(_slicing="exact")[1971.42]
	a.set(_slicing="nearest")[1971]
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
	    new.values = values.copy()
	    new.axes = self.axes.copy()

	return new
	#return Dimarray(self.values.copy(), self.axes.copy(), slicing=self.slicing, **{k:getattr(self,k) for k in self.ncattrs()})

    #
    # SLICING
    #

    @property
    def dims(self):
	""" axis names 
	"""
	return tuple([ax.name for ax in self.axes])

    def __getitem__(self, item): 
	""" get a slice (use xs method)
	"""
	items = np.index_exp[item] # tuple
    
	# dictionary <axis name> : <axis index> to feed into xs
	ix_nd = {self.axes[i].name: it for i, it in enumerate(items)}

	return self.xs(**ix_nd)

    def xs_axis(self, ix, axis=0, method=None, keepdims=False, **kwargs):
	""" cross-section or slice along a single axis

	input:
	    - ix    : index as axis value or integer position 
	    - axis  : int or str
	    - method: None (default), "nearest", "exact", "numpy"
	    - **kwargs: additional parameters passed to self.axes relative to slicing behaviour

	output:
	    - Dimarray object or python built-in type, consistently with numpy slicing

	>>> a.xs(45.5, axis=0)	 # doctest: +ELLIPSIS
	>>> a.xs(45.7, axis="lat") == a.xs(45.5, axis=0) # "nearest" matching
	True
	>>> a.xs(time=1952.5)
	>>> a.xs(time=70, method="numpy") # 70th element along the time dimension
	"""
	assert axis is not None, "axis= cannot be None in slicing"

	# get an axis object
	ax = self.axes[axis] # axis object
	axis_id = self.axes.index(ax) # corresponding integer index

	# get integer index/slice for axis valued index/slice
	if method is None:
	    method = self._slicing # slicing method

	# numpy-like indexing, do nothing
	if method == "numpy":
	    index = ix

	# otherwise locate the values
	else:
	    index = ax.loc(ix, method=method, **kwargs) 

	# make a numpy index  and use numpy's slice method (`slice(None)` :: `:`)
	index_nd = (slice(None),)*axis_id + (index,)
	newval = self.values[index_nd]
	newaxis = self.axes[axis][index] # returns an Axis object

	# if resulting dimension has reduced, remove the corresponding axis
	axes = copy.copy(self.axes)
	metadata = self._metadata.copy()

	# check for collapsed axis
	collapsed = not isinstance(newaxis, Axis)
	    
	# re-expand things even if the axis collapsed
	if collapsed and keepdims:

	    newaxis = Axis([newaxis], self.axes[axis].name) 
	    reduced_shape = list(self.shape)
	    reduced_shape[axis_id] = 1 # reduce to one
	    newval = np.reshape(newval, reduced_shape)

	    collapsed = False # set as not collapsed

	# If collapsed axis, just remove it and add new stamp
	if collapsed:
	    axes.remove(ax)
	    stamp = "{}={}".format(ax.name, newaxis)
	    append_stamp(metadata, stamp, inplace=True)

	# Otherwise just update the axis
	else:
	    axes[axis_id] = newaxis

	# If result is a numpy array, make it a Dimarray
	if isinstance(newval, np.ndarray):
	    result = self._constructor(newval, axes, **metadata)

	# otherwise, return scalar
	else:
	    result = newval

	return result

    def xs(self, ix=None, axis=0, method=None, keepdims=False, **axes):
	""" Cross-section, can be multidimensional

	input:

	    - ix : int or list or tuple or slice
	    - axis   : int or str
	    - method	: slicing method
	    => passed to xs_axis

	    - **axes  : provide axes as keyword arguments for multi-dimensional slicing
	    => chained call to xs_axis

	>>> a.xs(lon=(30.5, 60.5), lat=45.5) == a[:, 45.5, 30.5:60.5] # multi-indexing, slice...
	True
	>>> a.xs(time=1952, lon=-40, lat=70, method="nearest") # lookup nearest element (with bound checking)
	"""
	# single-axis slicing
	if ix is not None:
	    obj = self.xs_axis(ix, axis=axis, method=method, keepdims=keepdims)

	# multi-dimensional slicing <axis name> : <axis index value>
	# just a chained call
	else:
	    obj = self
	    for nm, idx in axes.iteritems():
		obj = obj.xs_axis(idx, axis=nm, method=method, keepdims=keepdims)

	return obj

    #
    # here some aliases to make things compatible with pandas
    #
    @property
    def loc(self):
	""" pandas-like: exact access to the index
	"""
	obj = self.copy(shallow=True)
	obj._slicing = 'exact'
	return obj

    @property
    def iloc(self):
	""" integer index-access
	"""
	obj = self.copy(shallow=True)
	obj._slicing = 'numpy'
	return obj

    @property
    def ix(self):
	""" automatic choice between numpy or axis-value indexing

	question: should follow this path?? this is error prone
	or rather make it an alias for iloc?
	"""
	if not self._pandas_like:
	    return self.iloc

	raise Warning("this method may disappear in the future")
	raise NotImplementedError('add the method to Locator')


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

    #
    # NUMPY TRANSFORMS
    #
    def apply(self, funcname, axis=0, skipna=True, args=(), **kwargs):
	""" apply along-axis numpy method

	parameters:
	----------
	    - funcname: numpy function name (str)
	    - axis    : int, str, tuple: axis or group of axes to apply the transform on
	    - skipna  : remove nans?
	    - args    : variable list of arguments before "axis"
	    - kwargs  : variable dict of keyword arguments after "axis"
	
	returns:
	--------
	    - Dimarray or scalar, consistently with ndarray behaviour

	NOTE: if an axis has weights, the `mean`, `std` and `var` will 
	      use these weights.
	"""
	assert type(funcname) is str, "can only provide function as a string"

	# If axis is provided as a tuple, apply the function on the collapsed array
	if type(axis) is tuple:
	    return self.collapse(*axis).apply(funcname, axis=0, skipna=skipna, args=args, **kwargs)

#	# Apply weighted transform if applicable
#	if funcname in ['mean','std','var'] and self.axes[axis].weights is not None:
#	    return self._apply_weighted(funcname, axis=axis, skipna=skipna, args=args, **kwargs)


	# Deal with NaNs
	values = self.values
	if skipna:

	    # replace with the optimized numpy function if existing
	    if funcname in ("sum","max","min","argmin","argmax"):
		funcname = "nan"+funcname

	    # otherwise convert to MaskedArray if needed
	    else:
		nans = np.isnan(values)
		if np.any(nans):
		    values = np.ma.array(values, mask=nans)

	# Get axis name and idx
	if axis is not None:
	    axis = self.axes.get_idx(axis)
	    axis_obj = self.axes[axis]
	    axis_nm = axis_obj.name

	kwargs['axis'] = axis
	result = getattr(values, funcname)(*args, **kwargs) 

	# Special treatment for argmax and argmin: return the corresponding index
	if funcname in ("argmax","argmin","nanargmax","nanargmin"):
	    assert axis is not None, "axis must not be None for "+funcname+", or apply on values"
	    result = axis_obj.values[result] # get actual axis values
	    return result

	# if scalar, just return it
	if not isinstance(result, np.ndarray):
	    return result

	# otherwise, fill NaNs back in
	if np.ma.isMaskedArray(result):
	    result = result.filled(np.nan)

	# New axes
	# ...nothing change for cumulative functions
	if funcname.startswith('cum'):
	    newaxes = self.axes.copy() # do not change anything

	# ...diff without collapsing dimension: add a first slice of NaNs to conserve axis size
	# ...and be bijective w.r.t. to reverse transform cumsum
	elif  funcname == "diff" and result.ndim == self.ndim: 
	    # first = (slice(None),)*axis + (0,) # first dimension along axis 
	    #nan_slice = np.empty_like(result[first]) # make a slice ...
	    nan_slice = np.empty_like(result.take([0], axis=axis)) # make a slice ...
	    nan_slice.fill(np.nan) # ...filled with NaNs
	    result = np.concatenate((nan_slice,result), axis=axis)
	    newaxes = self.axes.copy() # do not change anything

	else:
	    newaxes = [ax for ax in self.axes if ax.name != axis_nm]

	obj = self._constructor(result, newaxes)

	# add metadata back in
	obj._metadata_update_transform(self, funcname, axis_nm)

	return obj

    #
    # Add numpy transforms
    #
    mean = transform.NumpyDesc("apply", "mean")
    median = transform.NumpyDesc("apply", "median")
    diff = transform.NumpyDesc("apply", "diff")
    sum  = transform.NumpyDesc("apply", "sum")
    prod = transform.NumpyDesc("apply", "prod")

    cumsum = transform.NumpyDesc("apply", "cumsum")
    cumprod = transform.NumpyDesc("apply", "cumprod")

    min = transform.NumpyDesc("apply", "min")
    max = transform.NumpyDesc("apply", "max")
    ptp = transform.NumpyDesc("apply", "ptp")
    all = transform.NumpyDesc("apply", "all")
    any = transform.NumpyDesc("apply", "any")

    def weighted_mean(self, axis=0, skipna=True, weights=None):
	""" compute a weighted mean
	"""
	# use Dimarray future to get a N-D array of weights

	# standard case: guess weights
	if weights is None:
	    weights = self.get_weights()

	# user-provided: set NaNs where needed
	else:
	    weights.values[np.isnan(self.values)] = np.nan

	# Multiply values by the weights 
	sum_values = (self*weights).sum(axis=axis, skipna=skipna)
	sum_weights = (weights).sum(axis=axis, skipna=skipna)
	return sum_values / sum_weights

    def get_weights(self):
	""" get weight associated to the array
	"""
	if self.weights is not None:
	    weights = Dimarray(self.weights, *self.axes, **{'copy':False})

	else:
	    all_weights = [ax.get_weights().reshape(self.dims) for ax in self.axes]
	    weights = np.prod(all_weights, axis=0) # multiply the weights

	# fill NaNs in when necessary
	weights = weights.values[np.isnan(self.values)] = np.nan
	
	return weights

    def compress(self, condition, axis=None):
	""" analogous to numpy `compress` method
	"""
	return self.apply("compress", axis=axis, skipna=False, args=(condition,))

    def take(self, indices, axis=None):
	""" analogous to numpy `take` method
	"""
	return self.apply("take", axis=axis, skipna=False, args=(indices,))


#    #
#    # add an extra "where" method
#    #
#    def where(self, condition, otherwise=None, axis=None):
#	""" 
#	parameters:
#	-----------
#
#	    condition: bool array of same size as self, unless `axis=` is provided
#		 OR    `str` indicating a condition on axes
#	    otherwise: array of same size as self or scalar, replacement value when condition is False
#	    axis     : if provided, interpret the condition as applying along an axis
#
#	returns:
#	--------
#	    
#	    array with self values when condition is True, and `otherwise` if False
#	    if only `condition` is provided, return axis values for which `condition` is True
#
#	Examples:
#	---------
#	    a.where(a > 0)
#	"""
#	# convert scalar to the right shape
#	if np.size(otherwise) == 1:
#	    otherwise += np.zeros_like(self.values)
#
#	# evaluate str condition
#	if type(condition) is str:
#	    result = eval(condition, {ax.name:ax.values})
#
#	result = np.where(condition, [self.values, otherwise])


    #
    # RESHAPE THE ARRAY
    #
    def transpose(self, axes=None):
	""" Analogous to numpy, but also allows axis names

	>>> a = da.array(np.zeros((5,6)), lon=np.arange(6), lat=np.arange(5))
	>>> a          # doctest: +ELLIPSIS
	dimensions(5, 6): lat, lon
	array(...)
	>>> a.T         # doctest: +ELLIPSIS
	dimensions(6, 5): lon, lat
	array(...)
	>>> a.transpose([1,0]) == a.T == a.transpose(['lon','lat'])
	True
	"""
	if axes is None:
	    if self.ndim == 2:
		axes = [1,0] # numpy, 2-D case
	    elif self.ndim == 1:
		axes = [0]

	    else:
		raise ValueError("indicate axes value to transpose")

	# get equivalent indices 
	if type(axes[0]) is str:
	    axes = [self.dims.index(nm) for nm in axes]

	result = self.values.transpose(axes)
	newaxes = [self.axes[i] for i in axes]
	return self._constructor(result, newaxes)

    def collapse(self, *dims):
	""" collapse (or flatten) dimensions

	Input:
	    - *dims: variable list of axis names

	Output:
	    - Dimarray appropriately reshaped, with collapsed dimensions as first axis (tuples)

	This is useful to do a regional mean with missing values

	Note: can be passed via the "axis" parameter of the transformation, too

	Example:
	--------

	a.collapse('lon','lat').mean()

	Is equivalent to:

	a.mean(axis=('lon','lat')) 
	"""
	# First transpose the array so that the dimensions to collapse are at the front
	newdims = dims + tuple(nm for nm in self.dims if nm not in dims)	
	b = self.transpose(newdims) # dimensions to factorize in the front

	# Then collapse the first len(dims) dimensions in one new axis

	n = len(dims) # number of dimensions to collapse
	first_dim = np.sum(self.shape[:n])

	# Each element of the new first axis is a tuple (or a subarray of dimension n)
	axis_values = _expand(*[ax.values for ax in self.axes[:n]])

	# Combine the weights as a product of the individual axis weights
	axis_weights= _expand(*[ax.get_weights() for ax in self.axes[:n]]).prod(axis=1)

	if np.all(axis_weights == 1):
	    axis_weights = None

	assert first_dim == np.size(axis_values), "problem when reshaping"

	# Define the new axis
	first_axis = Axis(axis_values, name=dims, weights=axis_weights) 

	# Reshape the actual array values
	newshape = (first_dim,) + self.shape[n:]
	newvalues = b.values.reshape(newshape)

	# Define the new array
	new = self._constructor(newvalues, [first_axis,]+self.axes[n:], **self._metadata)

	return new


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
	    axes = [ax for i, ax in enumerate(self.axes) if self.shape[i] > 1]
	    res = self.values.squeeze()

	else:
	    axis = self.axes.get_idx(axis) 
	    axes = self.axes.copy()
	    if self.shape[axis] == 1:  # 0 is not removed by numpy...
		axes.pop(axis)
	    res = self.values.squeeze(axis)

	return self.__constructor(res, axes, **self._metadata)

    @property
    def T(self):
	return self.transpose()

    def newaxis(self, dim):
	""" add a new axis, ready to broadcast
	"""
	assert type(dim) is str, "dim must be string"
	values = self.values[np.newaxis] # newaxis is None 
	axis = Axis([None], dim) # create new dummy axis
	axes = self.axes.copy()
	axes.insert(0, axis)
	return self._constructor(values, axes, **self._metadata)

    def reshape(self, newdims):
	""" reshape an array according to a new set of dimensions

	Note: involves transposing the array
	"""
	# first append missing dimensions
	o = self
	for dim in newdims:
	    if dim not in self.dims:
		o = o.newaxis(dim)

	# now give it the right order
	return o.transpose(newdims)

    #
    # REINDEXING 
    #
 
    def reindex_axis(self, newaxis, axis=0, method='nearest'):
	""" reindex an array along an axis

	Input:
	    - newaxis: array or list on the new axis
	    - axis   : axis number or name
	    - method : "nearest", "exact", "interp"

	Output:
	    - Dimarray

	TO DO: optimize?
	"""
	if axis is None:
	    assert isinstance(newaxis, Axis), "provide name or an Axis object"
	    axis = newaxis.name

	if isinstance(newaxis, Axis):
	    newaxis = newaxis.values

	axis_id = self.axes.get_idx(axis)
	axis_nm = self.axes.get_idx(axis)
	ax = self.axes[axis_id] # Axis object

	# do nothing if axis is same or only None element
	if ax.values[0] is None or np.all(newaxis==ax.values):
	    return self

	# indices along which to sample
	if method in ("nearest","exact",None):

	    indices = np.empty(np.size(newaxis), dtype=int)
	    indices.fill(-1)

	    # locate elements one by one...
	    for i, val in enumerate(newaxis):
		try:
		    idx = ax.loc.locate(val, method=method)
		    indices[i] = idx

		except Exception, msg:
		    # not found, will be filled with Nans
		    pass

	    # prepare array to slice
	    values = self.values.take(indices, axis=axis_id)

	    # missing indices
	    # convert int to floats if necessary
	    if values.dtype == np.dtype(int):
		values = np.array(values, dtype=float, copy=False)

	    missing = (slice(None),)*axis_id + np.where(indices==-1)
	    #missing = _ndindex(np.where(indices==-1), axis_id)
	    values[missing] = np.nan # set missing values to NaN

	elif method == "interp":
	    raise NotImplementedError(method)

	else:
	    #return self.interp1d(newaxis, axis)
	    raise ValueError(method)

	# new Dimarray
	# ...replace the axis
	new_ax = Axis(newaxis, ax.name)
	axes = self.axes.copy()
	axes[axis_id] = new_ax # replace the new axis

	# ...initialize Dimarray
	obj = self._constructor(values, axes, **self._metadata)
	return obj


    def _reindex_axes(self, axes, method=None):
	""" reindex according to a list of axes
	"""
	obj = self
	newdims = [ax2.name for ax2 in axes]
	for ax in self.axes:
	    if ax.name in newdims:
		newaxis = axes[ax.name].values
		obj = obj.reindex_axis(newaxis, axis=ax.name, method=method)

	return obj

    def reindex_like(self, other, method=None):
	""" reindex like another axis

	note: only reindex axes which are present in other
	"""
	return self._reindex_axes(other.axes, method=method)

    def reindex(self, method=None, **kwds):
	""" reindex over several dimensions via keyword arguments, for convenience
	"""
	obj = self
	for k in kwds:
	    obj = obj.reindex_axis(kwds[k], axis=k, method=method)
	return obj

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
	return isinstance(other, Dimarray) and np.all(self.values == other.values) and self.axes == other.axes


    #
    # to behave like a dictionary w.r.t. first dimension (pandas-like)
    #
    def __iter__(self):
	for k in self.keys():
	    yield k

    def keys(self):
	return self.axes[0].values

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
	    val = self.xs_axis(k, axis=axis) # cross-section
	    yield k, val

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
    # export to other data types
    #

    def to_list(self):
	return [self[k] for k in self]

    def to_dict(self):
	""" return an ordered dictionary of sets
	"""
	from collect import Dataset
	d = Dataset()
	for v in self:
	    d[v] = self[v]
	return d

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
	a = la.larry(self.values, [ax.values for ax in self.axes])
	print "warning: dimension names have not been passed to larry"
	return a

    #
    #  I/O
    # 

    def save(self, f, name=None, *args, **kwargs):
	import ncio

	# add variable name if provided...
	if name is None and hasattr(self, "name"):
	    name = self.name

	ncio.write_variable(f, self, name, *args, **kwargs)

    @classmethod
    def load(cls, f, *args, **kwargs):
	import ncio
	return ncio.read_base(f, *args, **kwargs)

    #
    # Plotting
    #

    def plot(self, *args, **kwargs):
	""" by default, use pandas for plotting
	"""
	return self.to_pandas().plot(*args, **kwargs)

#
# Add a few transformations and plotting methods
#

# recursive application of obj => obj transformation
Dimarray.apply_recursive = transform.apply_recursive 

# 1D and bilinear interpolation (recursively applied)
Dimarray.interp1d = transform.interp1d_numpy
Dimarray.interp2d = transform.interp2d_mpl

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
	new = Dimarray.from_kwds(values, dims, **kwaxes) 


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
	o1, o2 = _align_objects(o1, o2)
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


def _align_objects(*objects):
    """ align dimensions of a list of objects by reindexing
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

	    objects[i] = o.reindex_axis(ax_values, axis=d)

    return objects

def _ndindex(indices, axis_id):
    """ return the N-D index from an along-axis index
    """
    return (slice(None),)*axis_id + np.index_exp[indices]

def _expand(*list_of_arrays):
    """ Expand a list of arrays ax1, ax2, ... to  a list of tuples [(ax1[0], ax2[0],..), (ax1[0], ax2[1]), ...]
    """
    kwargs = dict(indexing="ij")
    return np.array(zip(*np.meshgrid(*list_of_arrays, **kwargs)))
