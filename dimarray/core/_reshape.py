""" Module to regroup methods related to changing DimArray dimensions

Note: functions with self as first arguments are used as DimArray methods
"""
import numpy as np
from collections import OrderedDict
import functools

from axes import Axis, Axes, GroupedAxis
from tools import is_DimArray

#
# Broadcast
#
def broadcast(self, other):
    """ broadcast the array along a set of axes by repeating it as necessay

    other	     : DimArray or Axes objects or ordered Dictionary of axis values

    Examples:
    --------
    Create some dummy data:
    # ...create some dummy data:
    >>> lon = np.linspace(10, 30, 2)
    >>> lat = np.linspace(10, 50, 3)
    >>> time = np.arange(1950,1955)
    >>> ts = da.DimArray.from_kw(np.arange(5), time=time)
    >>> cube = da.DimArray.from_kw(np.zeros((3,2,5)), lon=lon, lat=lat, time=time)  # lat x lon x time
    >>> cube.axes  
    dimensions: 'lat', 'lon', 'time'
    0 / lat (3): 10.0 to 50.0
    1 / lon (2): 10.0 to 30.0
    2 / time (5): 1950 to 1954

    # ...broadcast timeseries to 3D data
    >>> ts3D = ts.broadcast(cube) #  lat x lon x time
    >>> ts3D
    dimarray: 30 non-null elements (0 null)
    dimensions: 'lat', 'lon', 'time'
    0 / lat (3): 10.0 to 50.0
    1 / lon (2): 10.0 to 30.0
    2 / time (5): 1950 to 1954
    array([[[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]],
    <BLANKLINE>
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]],
    <BLANKLINE>
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    """
    # Input as axes
    if isinstance(other, list):
	newaxes = other

    # Or as DimArray
    elif is_DimArray(other):
	newaxes = other.axes

    # Or as OrderedDict of axis names, axis values
    elif isinstance(other, OrderedDict):
	newaxes = [Axis(other[k], k) for k in other]

    else:
	raise TypeError("should be a DimArray, a list of Axis objects or an OrderedDict of Axis objects")

    if not isinstance(newaxes[0], Axis):
	raise TypeError("should be a DimArray, a list of Axis objects or an OrderedDict of Axis objects")

    newshape = [ax.name for ax in newaxes]

    # First give it the right shape
    newobj = self.reshape(newshape)

    # Then repeat along axes
    #for newaxis in newaxes:  
    for newaxis in reversed(newaxes):  # should be faster ( CHECK ) 
	if newobj.axes[newaxis.name].size == 1 and newaxis.size != 1:
	    newobj = newobj.repeat(newaxis.values, axis=newaxis.name)

    return newobj

#
# Transpose: permute dimensions
#
def transpose(self, axes=None):
    """ Permute dimensions
    
    Analogous to numpy, but also allows axis names
    >>> a = da.DimArray(np.zeros((2,3)), dims=['x0','x1'])
    >>> a          
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> a.T       
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x1', 'x0'
    0 / x1 (3): 0 to 2
    1 / x0 (2): 0 to 1
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.]])
    >>> np.all(a.transpose([1,0]) == a.T == a.transpose(['x1','x0']))
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

#
# Repeat the array along *existing* axis
#
def repeat(self, values, axis=None):
    """ expand the array along axis: analogous to numpy's repeat

    Signature: repeat(values=None, axis=None, **kwaxes)

    input:
	values  : integer (size of new axis) or ndarray (values  of new axis) 
	          or Axis object
	axis    : int or str (refer to the dimension along which to repeat)

	**kwaxes: alternatively, axes may be passed as keyword arguments 

    output:
	DimArray

    Sea Also:
    ---------
    newaxis

    Examples:
    --------
    >>> a = da.DimArray.from_kw(np.arange(3), time=[1950., 1951., 1952.])
    >>> a2d = a.reshape(('time','lon')) # lon is now singleton dimension
    >>> a2d.repeat([30., 50.], axis="lon")  
    dimarray: 6 non-null elements (0 null)
    dimensions: 'time', 'lon'
    0 / time (3): 1950.0 to 1952.0
    1 / lon (2): 30.0 to 50.0
    array([[0, 0],
	   [1, 1],
	   [2, 2]])
    """
    # default axis values: 0, 1, 2, ...
    if type(values) is int:
	values = np.arange(values)

    if axis is None:
	assert hasattr(values, "name"), "must provide axis name or position !"
	axis = values.name

    # Axis position and name
    idx, name = self._get_axis_info(axis) 

    if name not in self.dims:
	raise ValueError("can only repeat existing axis, need to reshape first (or use broadcast)")

    if self.axes[idx].size != 1:
	raise ValueError("can only repeat singleton axes")

    # Numpy reshape: does the check
    newvalues = self.values.repeat(np.size(values), idx)

    # Create the new axis object
    if not isinstance(values, Axis):
	newaxis = Axis(values, name)

    else:
	newaxis = values

    # New axes
    newaxes = [ax for ax in self.axes]
    newaxes[idx] = newaxis

    # Update values and axes
    return self._constructor(newvalues, newaxes, **self._metadata)


def newaxis(self, name, values=None, pos=0):
    """ add a new axis, ready to broadcast

    name: `str`, axis name
    values: optional, if provided, broadcast the array along the new axis
	    call `repeat(name, values)` after inserting the new singleton 
	    dimension (see `repeat`) for more information.
    pos: `int`, optional: axis position, default 0 (first axis)

    Examples:
    ---------
    >>> a = DimArray([1,2])
    >>> a
    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (2): 0 to 1
    array([1, 2])
    >>> a.newaxis('new', pos=1)
    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0', 'new'
    0 / x0 (2): 0 to 1
    1 / new (1): None to None
    array([[1],
	   [2]])
    >>> a.newaxis('new', values=['a','b'],pos=1)
    dimarray: 4 non-null elements (0 null)
    dimensions: 'x0', 'new'
    0 / x0 (2): 0 to 1
    1 / new (2): a to b
    array([[1, 1],
	   [2, 2]])
    """
    assert type(name) is str, "name must be string"
    if name in self.dims:
	raise ValueError("dimension already present: "+name)

    assert type(pos) is int
    if pos == -1: pos = len(self.dims)

    newaxis = (slice(None),)*pos + (np.newaxis,) # pad with ":" to match pos
    newvalues = self.values[newaxis] 

    # create new dummy axis
    axis = Axis([None], name) 

    # insert new axis
    axes = self.axes.copy()
    axes.insert(pos, axis)

    # create new object
    newobj = self._constructor(newvalues, axes, **self._metadata)

    # If values is provided, repeat the array along values
    if values is not None:
	newobj = newobj.repeat(values, axis=pos)

    return newobj

#
# Remove / add singleton axis
#
def squeeze(self, axis=None):
    """ Analogous to numpy, but also allows axis name

    >>> a = da.DimArray([[[1,2,3]]])
    >>> a
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0', 'x1', 'x2'
    0 / x0 (1): 0 to 0
    1 / x1 (1): 0 to 0
    2 / x2 (3): 0 to 2
    array([[[1, 2, 3]]])
    >>> a.squeeze()
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x2'
    0 / x2 (3): 0 to 2
    array([1, 2, 3])
    >>> a.squeeze(axis='x1')
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0', 'x2'
    0 / x0 (1): 0 to 0
    1 / x2 (3): 0 to 2
    array([[1, 2, 3]])
    """
    if axis is None:
	newaxes = [ax for ax in self.axes if ax.size != 1]
	res = self.values.squeeze()

    else:
	idx, name = self._get_axis_info(axis) 
	res = self.values.squeeze(idx)
	newaxes = [ax for ax in self.axes if ax.name != name or ax.size != 1] 

    return self._constructor(res, newaxes, **self._metadata)

#
# Reshape by adding/removing as many singleton axes as needed to match prescribed dimensions
#
def reshape(self, newdims):
    """ Conform array to new dimensions
    
    input:
	newdims: tuple or list of dimensions (`str`)

    output:
	reshape: DimArray with reshape.dims == tuple(newdims)
    
    Method: add/remove singleton dimensions and transpose array

    Examples:
    ---------
    >>> a = DimArray([7,8])
    >>> a
    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (2): 0 to 1
    array([7, 8])
    >>> a.reshape(('x0','new'))
    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0', 'new'
    0 / x0 (2): 0 to 1
    1 / new (1): None to None
    array([[7],
           [8]])
    """
    # Do nothing if dimensions already match
    if tuple(newdims) == self.dims:
	return self

    # Remove unwanted singleton dimensions, if any
    o = self
    for dim in o.dims:
	assert type(dim) is str, "newdims must be a tuple of axis names (`str`)"
	if dim not in newdims:
	    o = o.squeeze(dim)

    # Transpose array to match existing dimensions
    o = o.transpose([dim for dim in newdims if dim in o.dims])

    # Add missing dimensions by inserting singleton axes
    for i, dim in enumerate(newdims):
	if dim not in o.dims:
	    o = o.newaxis(dim, pos=i)

    return o


#
# Group/ungroup subsets of axes to perform operations on partly flattened array
#
def group(self, dims, exclude=False, insert=0):
    """ group (or flatten) a subset of dimensions

    Input:
	- dims: list or tuple of axis names
	- exclude [False]: if True, dims are interpreted as the dimensions to exclude
	    and all the other dimensions are grouped
	- insert: position where to insert the grouped axis 
		  (by default, any grouped dimensions is placed as first axis)

    Output:
	- DimArray appropriately reshaped, with collapsed dimensions as first axis (tuples)

    This is useful to do a regional mean with missing values

    Note: can be passed via the "axis" parameter of the transformation, too

    Example:
    --------

    a.group(('lon','lat')).mean()

    Is equivalent to:

    a.mean(axis=('lon','lat')) 
    """
    if type(dims) not in (tuple, list, set):
	raise TypeError("dimensions to group must be a list or a tuple  or a set")

    # exclude? mirror call
    assert type(exclude) is bool, "exclude must be a boolean !"
    if exclude:
	dims = [d for d in self.dims if d not in dims]

    # check the 
    if not set(dims).issubset(self.dims):
	raise ValueError("dimensions to group must be a subset of existing dimensions")

#    # if 
#    if len(dims) == 1:
#	if self.dims[insert] != dims[0]:
#	    newdims = [d for d in self.dims if d != dims[0]]
#	    newdims = newdims[:insert] + list(dims) + newdims[insert:]
#	    self = self.transpose(newdims)
#	return self
#	#raise ValueError("cannot group less than 2 dimensions")

    # make sure we have a tuple of strings
    dims = [self.axes[d].name for d in dims]

    # reorder dimensions to group and convert to tuple
    dims = [d for d in self.dims if d in dims]
    dims = tuple(dims)

    n = len(dims) # number of dimensions to group
    ii = self.dims.index(dims[0]) # start

    # dimension to insert the new axis at
    if insert is None: 
	insert = ii  # by default, do not reshape

    # If dimensions do not follow each other, transpose first
    if dims != self.dims[insert:insert+len(dims)]:

	# new array shape, assuming dimensions are properly aligned
	newdims = [ax.name for ax in self.axes if ax.name not in dims]
	newdims = newdims[:insert] + list(dims) + newdims[insert:]
    
	b = self.transpose(newdims) # dimensions to factorize in the front

	return b.group(dims, insert=insert)

    # Create a new grouped axis
    newaxis = GroupedAxis(*[ax for ax in self.axes if ax.name in dims])

    # New axes
    newaxes = [ax for ax in self.axes if ax.name not in dims]
    newaxes.insert(insert, newaxis)

    # Reshape the actual array values
    newshape = [ax.size for ax in newaxes]
    newvalues = self.values.reshape(newshape)

    # Define the new array
    new = self._constructor(newvalues, newaxes, **self._metadata)

    return new

def ungroup(self, axis=None):
    """ opposite from group

    axis: axis to ungroup as int or str (default: ungroup all)
    """
    # by default, ungroup all
    if axis is None:
	grouped_axes = [ax.name for ax in self.axes if isinstance(ax, GroupedAxis)]
	obj = self
	for axis in grouped_axes:
	    obj = obj.ungroup(axis=axis)
	return obj

    assert type(axis) in (str, int), "axis must be integer or string"

    group = self.axes[axis]    # axis to expand
    axis = self.dims.index(group.name) # make axis be an integer

    assert isinstance(group, GroupedAxis), "can only ungroup a GroupedAxis"

    newshape = self.shape[:axis] + tuple(ax.size for ax in group.axes) + self.shape[axis+1:]
    newvalues = self.values.reshape(newshape)
    newaxes = self.axes[:axis] + group.axes + self.axes[axis+1:]

    return self._constructor(newvalues, newaxes, **self._metadata)

#
# groupby method
#

class Desc(object):
    """ descriptor to add methods with pre-selected axis
    """
    def __init__(self, nm, *args, **kwargs):
	self.nm = nm
	self.args = args
	self.kwargs = kwargs

    def __get__(self, obj, cls=None):
	"""
	"""
	method = getattr(obj.a, self.nm)

	if callable(method):
	    method = functools.partial(method, *self.args, **self.kwargs)
	#if self.axis is not None:
	#    method functools.partial(method, axis=self.axis)

	return method

class GroupBy(object):
    """ Make it easy to display stats for one variable
    """
    def __init__(self, a, dims):
	"""
	"""
	self.a = a
	self.dims = dims


    values = Desc('values')

    __getitem__ = Desc('__getitem__')
    ix = Desc('ix')

    take = Desc('take', axis=0)

    mean = Desc('mean', axis=-1)
    std = Desc('std', axis=-1)
    var = Desc('var', axis=-1)
    median = Desc('median', axis=-1)
    sum = Desc('sum', axis=-1)
    prod = Desc('prod', axis=-1)
    all = Desc('all', axis=-1)
    any = Desc('any', axis=-1)
    min = Desc('min', axis=-1)
    max = Desc('max', axis=-1)
    ptp = Desc('ptp', axis=-1)

#_trans = 'mean','std','var','median','sum','prod', 'all','any','min','max','ptp'
#for op in _trans:
#    GroupBy.__dict__[op] = Desc(op, axis=-1)

def groupby(self, *dims):
    """ group by one or several variables along which stat functions can be applied

    parameters:
	*dims: variable list of dims to keep, all others are flattened

    returns:
	GroupBy object

    Examples:
    ---------
    >>> a = DimArray([[1,2,3],[4,5,6]], [('x',[1,2]), ('items',['a','b','c'])])
    >>> a = a.newaxis('y',4) # add an axis of length 4
    >>> a.groupby('items').mean()
    dimarray: 3 non-null elements (0 null)
    dimensions: 'items'
    0 / items (3): a to c
    array([ 2.5,  3.5,  4.5])
    """
    obj = group(self, dims, exclude=True, insert=0).T
    return GroupBy(obj, dims)
