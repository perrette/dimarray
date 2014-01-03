""" Module to regroup methods related to changing DimArray dimensions

Note: functions with self as first arguments are used as DimArray methods
"""
import numpy as np
from collections import OrderedDict

from axes import Axis, Axes, GroupedAxis
from dimarray.lib.tools import is_DimArray
from dimarray.lib.decorators import axes_as_keywords

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
    >>> a = da.DimArray.from_kw(np.arange(2), time=[1950., 1951., 1952.])
    >>> a2d = a.reshape(('time','lon')) # lon is now singleton dimension
    >>> a2d.repeat([30., 50.], axis="lon")  
    dimarray: 10 non-null elements (0 null)
    dimensions: 'time', 'lon'
    0 / time (5): 1950 to 1954
    1 / lon (2): 30.0 to 40.0
    array([[0, 1],
	   [0, 1],
	   [0, 1],
	   [0, 1],
	   [0, 1]])
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

    >>> a = da.DimArray(np.zeros((5,6)), lon=np.arange(6), lat=np.arange(5))
    >>> b = a.take([0], axis='lat')
    >>> b
    dimensions(1, 6): lat, lon
    array([[ 0.,  1.,  2.,  3.,  4.,  5.]])
    >>> b.squeeze()
    dimensions(6,): lon
    array([ 0.,  1.,  2.,  3.,  4.,  5.])
    """
    if axis is None:
	newaxes = [ax for ax in self.axes if ax.size != 1]
	res = self.values.squeeze()

    else:
	idx, name = self._get_axis_info(axis) 
	res = self.values.squeeze(idx)
	newaxes = [ax for ax in self.axes if ax.name != name or ax.size != 1] 

    return self.__constructor(res, newaxes, **self._metadata)

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
    >>> a = DimArray([[7,8]])
    >>> a
    >>> a.reshape(('x0','new'))
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

def group(self, dims, keep=False, insert=None):
    """ group (or flatten) a subset of dimensions

    Input:
	- *dims: variable list of axis names
	- keep [False]: if True, dims are interpreted as the dimensions to keep (mirror)
	- insert: position where to insert the grouped axis 
		  (by default, just reshape in the numpy sense if possible, 
		  otherwise insert at the position of the first dimension to group)

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
    assert type(keep) is bool, "keep must be a boolean !"
    assert type(dims[0]) is str, "dims must be strings"

    # keep? mirror call
    if keep:
	dims = tuple(d for d in self.dims if d not in dims)

    n = len(dims) # number of dimensions to group
    ii = self.dims.index(dims[0]) # start
    if insert is None: 
	insert = ii

    # If dimensions do not follow each other, have all dimensions as first axis
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

