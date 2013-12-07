""" Array with dimensions
"""
import numpy as np
import copy

import plotting

from metadata import Variable, Metadata
from axes import Axis, Axes
from lazyapi import pandas_obj

use_pandas = True # use pandas to align data

__all__ = []

#
# Main class:
#
class DimArray(Variable):
    """ numpy's ndarray with physical dimensions

    * numpy's ndarrays with named axes and metadata organized on the netCDF format.
    * follow pandas' api but works for any dimension and any axis name
    * includes most numpy transformations
    * alias method for matplotlib plotting.
    * natural netCDF I/O
    """
    _order = None  # set a general ordering relationship

    # pre-defined metadata (see doc in metadata.py)
    name = Metadata("name","")    # variable names
    units = Metadata("units","")  # variable units
    descr = Metadata("descr","")  # variable description
    stamp = Metadata("stamp","")  # record dimensions during slicing and transformations

    #
    # also protect the axes attribute
    #
    @property 
    def axes(self):
	return self._axes

    @values.setter
    def axes(self, newval):
	if np.shape(newval) != np.shape(self._values):
	    msg = "To reshape dimensions try one of the exising methods "
	    msg += "(squeeze, transpose...) or create a new <{cls}> instance".format(cls=self.__class__.__name__)
	    raise ValueError(msg)

    #
    # NOW MAIN BODY OF THE CLASS
    #

    def __init__(self, values, axes=None, dtype=None, _slicing=None, copy=False, **attrs):
	""" Initialization

	values	: numpy-like array, or DimArray instance
	axes	: Axes instance, or list of tuples or list of Axis objects
		  This argument can only be omitted if values is an instance of DimArray
		    
	**attrs: metadata 

	dtype, copy: passed to np.array()

	_slicing: default slicing method (mostly for internal use)
	"""
	#
	# array values
	#
	avalues = np.array(values, dtype=dtype, copy=copy)

	#
	# Initialize the axes
	#
	if axes is None:
	    assert hasattr(values, "axes"), "need to provide axes (as Axes object or tuples)!"
	    axes = values.axes

	assert isinstance(axes, list),  "axes must be passed as a list"

	if len(axes) == 0:
	    axes = Axes()

	elif type(axes[0]) is tuple:
	    axes = Axes.from_tuples(*axes)

	elif type(axes[0]) is Axis:
	    axes = Axes(axes)

	assert isinstance(axes, Axes), "axes must be a list of tuples or an Axes instance"

	#
	# store all fields
	#

	self.values = avalues
	self.axes = axes

	# option
	self._slicing = _slicing

	# attributes
	self.attrs = attrs

	# Define attributes
	_check_attrs(attrs) # check whether each attribue conform, raise Exception if not
	self.__dict__.update(attrs) # passing keyword arguments

	# exact shape
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
    def __constructor(values, axes, **kwargs):
	""" Internal API for the constructor: check whether a pre-defined class exists

	values	: numpy-like array
	axes	: Axes instance 
	**kwargs: additional parameters

	This static method is used whenever a new DimArray needs to be instantiated
	for example after a transformation.

	This makes the sub-classing process easier since only this method needs to be 
	overloaded to make the sub-class a "closed" class.
	"""
	assert isinstance(axes, Axes), "Need to provide an Axes object !"

	# look whether a particular pre-defined array matches the dimensions
	import defarray

	# the dimensions for the object
	dims = axes.names

	# loop over all variables defined in defarray
	cls = None
	for obj in vars(defarray): 
	    if isinstance(obj, defarray.Defarray):
		if tuple(dims) == cls._dimensions:
		    cls = obj

	# initialize with the specialized class
	if cls is not None:
	    new = cls(values, *axes, **kwargs)

	# or just with the normal class constructor
	else:
	    #new = dimarray(values, axes)
	    new = DimArray(values, axes, **kwargs)
	    #new.set(inplace=True, **kwargs)
	    return new

    @classmethod
    def from_tuples(cls, values, *args, **kwargs):
	""" initialize DimArray with with variable list of tuples, and attributes

	values	    : numpy-like array (passed to array())
	*args	    : variable list of tuples to define axes: ('x0',val0), ('x1',val1), ...
	**kwargs    : list of tuples
	"""
	axes = Axes.from_tuples(*args)
	new = cls(values, axes, **kwargs)
	return new


    @classmethod
    def from_list(cls, values, axes, names=None, **kwargs):
	""" initialize DimArray with with variable list of tuples, and attributes

	values	    : numpy-like array (passed to array())
	*args	    : variable list of tuples to define axes: ('x0',val0), ('x1',val1), ...
	**kwargs    : list of tuples
	"""
	axes = Axes.from_list(axes, names)
	new = cls(values, axes, **kwargs)
	return new

    @classmethod
    def from_dict(cls, values, names=None, **kwargs):
	""" initialize DimArray with with variable list of tuples, and attributes

	values	    : numpy-like array (passed to array())
	*args	    : variable list of tuples to define axes: ('x0',val0), ('x1',val1), ...
	**kwargs    : list of tuples
	"""
	if len(kwargs) > 0:
	    for k in kwargs:
		if type(kwargs[k]) is str:
		    print "PASSED:",k,":",kwargs[k]
		    msg = \
""" no attribute can be passed with the from_dict method 
==> try using the set() method instead, e.g. DimArray.from_dict(values, **kwargs).set(val1=val2)
"""
		    raise ValueError(msg)

	    axes = Axes.from_dict(shape=np.shape(values), names=names, **kwargs)
	return cls(values, axes)

    #
    # Attributes access
    #
    def __getattr__(self, att):
	""" return dimension or numpy attribue
	"""
	# check for dimensions
	if att in self.axes.names:
	    ax = self.axes[att]
	    return ax.values # return numpy array

	# numpy attributes: size, shape etc...
	elif hasattr(self.values, att): # and not inspect.isfunction( getattr(self.values, att) ):
	    return getattr(self.values, att)

	else:
	    raise ValueError("unknown attribute: "+att)

    def copy(self, shallow=False):
	""" copy of the object and update arguments

	shallow: if True, does not copy values and axes
	"""
	import copy
	new = copy.copy(self) # shallow copy

	if not shallow:
	    new.set(values=values.copy(), axes=self.axes.copy(), inplace=True) 

	return new
	#return DimArray(self.values.copy(), self.axes.copy(), slicing=self.slicing, **{k:getattr(self,k) for k in self.ncattrs()})

    #
    # Slicing
    #

    @property
    def dims(self):
	""" axis names :: axes.names
	"""
	return tuple([ax.name for ax in self.axes])

    def _get_axis_idx(self, axis):
	""" always return an integer axis
	"""
	if type(axis) is str:
	    axis = self.dims.index(axis)

	return axis


    def __getitem__(self, item): 
	""" get a slice (use xs method)
	"""
	items = np.index_exp[item] # tuple
    
	# dictionary <axis name> : <axis index> to feed into xs
	slice_nd = {self.axes[i].name: it for i, it in enumerate(items)]

	return self.xs(**slice_nd)


    def xs_axis(self, slice_, axis=0, method=None, **kwargs):
	""" slice a long a single axis

	input:
	    - slice_: the slice
	    - axis  : int or str
	    - method: None (default), nearest, exact, numpy
	    - **kwargs: additional parameters passed to self.axes relative to slicing behaviour

	output:
	    - DimArray object

	>>> a.xs(45.5, axis=0)	 # doctest: +ELLIPSIS
	>>> a.xs(45.7, axis="lat") == a.xs(45.5, axis=0) # "nearest" matching
	True
	>>> a.xs(time=1952.5)
	>>> a.xs(time=70, method="numpy") # 70th element along the time dimension
	"""
	assert axis is not None, "axis= cannot be None in slicing"

	# get an axis object
	axis_obj = self.axes[axis] # axis object
	axis_id = self.axes.index(axis_obj) # corresponding integer index

	# get integer index/slice for axis valued index/slice
	index = axis.loc(slice_, method=method, **kwargs) 

	# make a numpy index  and use numpy's slice method (`slice(None)` :: `:`)
	index = (slice(None),)*(axis_id-1) + (index,)
	newval = self.values[index]

	# if resulting dimension has reduced, remove the corresponding axis
	if newval.ndim < self.ndim:
	    axes = copy.copy(self.axes)
	    axes.remove(axis_obj)
	    attrs = {axis_id.name:axis.values[index]} # add attribute

	return self.__constructor(newval, axes, **attrs)

    def xs(self, slice_=None, axis=0, method=None, **axes):
	""" Cross-section, can be multidimensional

	input:

	    - slice_ : int or list or tuple or slice
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
	if slice_ is not None:
	    obj = self.xs_axis(slice_, axis=axis, method=method)

	# multi-dimensional slicing <axis name> : <axis index value>
	# just a chained call
	else:
	    obj = self
	    for nm, idx in axes.iteritems():
		obj = obj.xs_axis(idx, axis=nm, method=method)

	return obj

    #
    # here some aliases to make things compatible with pandas
    #

    @property
    def ix(self)
	""" just an alias to the default behaviour
	"""
	raise NotImplementedError("pandas-like convenience' ix access is not yet implemented, \
		main methods are standard top-level `[]` syntax for axis-value-based indexing \
		(works in N-D) and iloc instead for integer indexing (standard numpy). `loc` \
		is also provided as a convenience to do pandas' exact-match indexing for float-
		valued array instead of nearest match, the default here.")

	return self

    @property
    def loc(self):
	""" pandas-like: exact access to the index
	"""
	return self.set(_slicing='exact', inplace=False)

    @property
    def iloc(self):
	""" integer index-access
	"""
	return self.set(_slicing='numpy', inplace=False)

    def __eq__(self, other): 
	return isinstance(other, DimArray) and np.all(self.values == other.values) and self.axes == other.axes

    #
    # to behave like a dictionary w.r.t. first dimension (pandas-like)
    #
    def __iter__(self):
	for k in self.keys():
	    yield k

    def keys(self):
	return self.axes[0].values

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

	if self.size < 10:
	    line = "dimarray: "+repr(self.values)
	else:
	    line = "dimarray: {} non-null elements ({} null)".format(nonnull, self.size-nonnull)
	lines.append(line)

	if np.any([getattr(self,k) is not "" for k in self.ncattrs()]):
	    line = ", ".join(['%s: "%s"' % (att, getattr(self, att)) for att in self.ncattrs()])
	    lines.append(line)

	if self.size > 1:
	    line = repr(self.axes)
	    lines.append(line)

	return "\n".join(lines)


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
	if axes is not None:
	    axes = [self._get_axis_idx(ax) for ax in axes]
	    names = [self.names[i] for i in axes]

	else:
	    names = [self.names[i] for i in 1,0] # work only 2D

	result = self.values.transpose(axes)
	return self.__constructor(result, names=names)



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

	# make sure the axis has the right name
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

    def save(self, f, *args, **kwargs):
	import ncio

	# add variable name if provided...
	if len(args) == 0 and 'name' not in kwargs:
	    if self.name:
		args = [self.name]+list(args)

	ncio.write_variable(f, self, *args, **kwargs)

    @classmethod
    def load(cls, f, *args, **kwargs):
	import ncio
	return ncio.read_base(f, *args, **kwargs)

    def squeeze(self, axis=None):
	""" reduce singleton dimensions
	"""
	obj = super(DimArray, self).squeeze(axis=axis)
	axes = Axes([self.axes[nm] for nm in obj.names])
	return __constructor(obj.values, axes)

    #
    # Plotting
    #

    def plot(self, *args, **kwargs):
	""" by default, use pandas for plotting
	"""
	return self.to_pandas().plot(*args, **kwargs)

    plot = plotting.plot
    contourf = plotting.contourf



def dimarray(values, axes=None, names=None, name="", units="", descr="", dtype=None, _slicing=None, **kwargs):
    """ Wrapper for initialization
    """
    if axes is None and hasattr(values, "axes"):
	axes = values.axes

    if isinstance(axes, np.ndarray):
	if np.ndim(values) == 1:
	    axes = [axes]
	    if type(names) is str: names = [names]
	else:
	    print axes
	    raise ValueError("axes must be passed as a list")

    if len(kwargs) > 0:
	new = DimArray.from_dict(values, axes, **kwargs) 

    # scalar values
    elif len(axes) == 0:
	new = DimArray(values, axes) 

    elif type(axes[0]) is tuple:
	new = DimArray.from_tuples(values, *axes) 

    else:
        new = DimArray.from_list(values, axes, names=names) 

    # reserved attributes
    new.set(name=name, units=units, descr=descr, _slicing=_slicing, inplace=True) 

    return new
