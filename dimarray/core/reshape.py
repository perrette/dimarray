""" Module to regroup methods related to changing DimArray dimensions

Note: functions with self as first arguments are used as DimArray methods
"""
import numpy as np
from collections import OrderedDict
import functools

from axes import Axis, Axes, MultiAxis
#from dimarray.tools import is_DimArray
import dimarray as da
from dimarray.tools import deprecated_func

#
# Broadcast
#
def broadcast(self, other):
    """ repeat array to match target dimensions

    Parameters
    ----------
    other : DimArray or Axes objects or ordered Dictionary of axis values

    Returns
    -------
    DimArray

    Examples
    --------
    Create some dummy data:
    # ...create some dummy data:

    >>> import dimarray as da
    >>> lon = np.linspace(10, 30, 2)
    >>> lat = np.linspace(10, 50, 3)
    >>> time = np.arange(1950,1955)
    >>> ts = da.DimArray(np.arange(5), axes=[time], dims=['time'])
    >>> cube = da.DimArray(np.zeros((3,2,5)), axes=[('lat',lat), ('lon',lon), ('time',time)])  # lat x lon x time
    >>> cube.axes  
    0 / lat (3): 10.0 to 50.0
    1 / lon (2): 10.0 to 30.0
    2 / time (5): 1950 to 1954

    # ...broadcast timeseries to 3D data

    >>> ts3D = ts.broadcast(cube) #  lat x lon x time
    >>> ts3D
    dimarray: 30 non-null elements (0 null)
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
    elif isinstance(other, da.DimArray):
         newaxes = other.axes

    # Or as OrderedDict of axis names, axis values
    elif isinstance(other, OrderedDict):
        newaxes = [Axis(other[k], k) for k in other]

    else:
        raise TypeError("should be a DimArray, a list of Axis objects or an OrderedDict of Axis objects")

    if len(newaxes) > 0 and not isinstance(newaxes[0], Axis): # just check the first as basic test
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
def transpose(self, *dims):
    """ Permute dimensions
    
    Analogous to numpy, but also allows axis names

    Parameters
    ----------
    *dims : int or str
        variable list of dimensions

    Returns
    -------
    transposed_array : DimArray

    See also
    ---------
    reshape, flatten, unflatten, newaxis

    Examples
    --------
    >>> import dimarray as da
    >>> a = da.DimArray(np.zeros((2,3)), ['x0','x1'])
    >>> a          
    dimarray: 6 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> a.T       
    dimarray: 6 non-null elements (0 null)
    0 / x1 (3): 0 to 2
    1 / x0 (2): 0 to 1
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.]])
    >>> (a.T == a.transpose(1,0)).all() and (a.T == a.transpose('x1','x0')).all()
    True
    """
    if len(dims) == 1 and type(dims[0]) in (list, tuple, set):
        dims = dims[0]

    if len(dims) == 0:
        if self.ndim == 2:
            dims = [1,0] # numpy, 2-D case
        elif self.ndim == 1:
            dims = [0]
        elif self.ndim == 0:
            return self
        else:
            raise ValueError("indicate dimensions to transpose")

    # get equivalent indices 
    newshape, _ = self._get_axes_info(dims)

    result = self.values.transpose(newshape)
    newaxes = [self.axes[i] for i in newshape]
    return self._constructor(result, newaxes, **self.attrs)

def swapaxes(self, axis1, axis2):
    """ Swap two axes
    
    analogous to numpy's swapaxes, but can provide axes by name

    Parameters
    ----------
    axis1, axis2 : int or str
        axes to swap (transpose)

    Returns
    -------
    transposed_array : DimArray

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray(np.arange(2*3*4).reshape(2,3,4))
    >>> a.dims
    ('x0', 'x1', 'x2')
    >>> b = a.swapaxes('x2',0) # put 'x2' at the first position
    >>> b.dims
    ('x2', 'x1', 'x0')
    >>> b.shape
    (4, 3, 2)
    """
    pos, _ = self._get_axes_info([axis1, axis2])
    axis1, axis2 = pos  # axis positions
    newshape = []
    for i in range(self.ndim):
        if i == axis1:
            newshape.append(axis2)
        elif i == axis2:
            newshape.append(axis1)
        else:
            newshape.append(i)
    return transpose(self, newshape)

def rollaxis(self, axis, start=0):
    """ Roll the specified axis backwards, until it lies in a given position.

    Parameters
    ----------
    axis : int or str
        The axis to roll backwards.  The positions of the other axes do not
        change relative to one another.
    start : int, optional
        The axis is rolled until it lies before this position.  The default,
        0, results in a "complete" roll.

    Returns
    -------
    res : DimArray instance
        Output array.

    See Also
    --------
    roll : Roll the elements of an array by a number of positions along a
        given axis.  (TODO)

    Examples
    --------
    >>> import dimarray as da
    >>> a = da.ones(shape=(3,4,5,6))
    >>> a.rollaxis(3, 1).shape
    (3, 6, 4, 5)
    >>> a.rollaxis(2).shape
    (5, 3, 4, 6)
    >>> a.rollaxis(1, 4).shape
    (3, 5, 6, 4)
    """
    axis, _ = self._get_axis_info(axis) # position

    # use numpy rollaxis on an empty array to determine the shape
    fake = np.ones(range(self.ndim)) # first is 0 ==> empty
    newshape = np.rollaxis(fake, axis, start).shape

    return self.transpose(newshape)

#
# Repeat the array along *existing* axis
#
def repeat(self, values, axis=None):
    """ expand the array along an existing axis
    
    Parameters
    ----------
    values : int or ndarray or Axis instance
        int: size of new axis
        ndarray: values  of new axis 
    axis : int or str 
        refer to the dimension along which to repeat

    **kwaxes : key-word arguments
        alternatively, axes may be passed as keyword arguments 

    Returns
    -------
    DimArray

    See Also
    --------
    newaxis

    Examples
    --------
    >>> import dimarray as da
    >>> a = da.DimArray(np.arange(3), labels = [[1950., 1951., 1952.]], dims=('time',))
    >>> a2d = a.newaxis('lon', pos=1) # lon is now singleton dimension

    >>> a2d.repeat(2, axis="lon")  
    dimarray: 6 non-null elements (0 null)
    0 / time (3): 1950.0 to 1952.0
    1 / lon (2): 0 to 1
    array([[0, 0],
           [1, 1],
           [2, 2]])

    >>> a2d.repeat([30., 50.], axis="lon")  
    dimarray: 6 non-null elements (0 null)
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
    return self._constructor(newvalues, newaxes, **self.attrs)


def newaxis(self, name, values=None, pos=0):
    """ add a new axis

    Add a singleton axis to ease broadcasting, 
    and repeat array along this axis if required.

    Parameters
    ----------
    name : str
        axis name
    values: array-like, optional
            if provided, broadcast the array along the new axis
            call `repeat(name, values)` after inserting the new singleton 
            dimension (see `repeat`) for more information.
    pos : int, optional
        axis position, default 0 (first axis)

    Returns
    -------
    DimArray

    Notes
    -----
    Numpy provides a np.newaxis constant (equal to None), to augment the array 
    dimensions with new singleton axes. In dimarray, newaxis has been 
    implemented as an array method, which requires to indicate axis name and 
    optionally axis position (`pos=`). 
    Additionally, passing providing values will repeat the array along that 
    dimension as many times as necessary.

    See Also
    --------
    squeeze, repeat

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray([1,2])
    >>> a
    dimarray: 2 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    array([1, 2])
    >>> a.newaxis('new', pos=1)
    dimarray: 2 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / new (1): None to None
    array([[1],
           [2]])
    >>> a.newaxis('new', values=['a','b'],pos=1)
    dimarray: 4 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / new (2): 'a' to 'b'
    array([[1, 1],
           [2, 2]])
    """
    assert isinstance(name, basestring), "name must be string"
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
    newobj = self._constructor(newvalues, axes, **self.attrs)

    # If values is provided, repeat the array along values
    if values is not None:
        newobj = newobj.repeat(values, axis=pos)

    return newobj

#
# Remove / add singleton axis
#
def squeeze(self, axis=None):
    """ Squeeze singleton axes
    
    Analogous to numpy, but also allows axis name

    Parameters
    ----------
    axis : int or str or None 
        axis to squeeze
        default is None, to remove all singleton axes

    Returns
    -------
    squeezed_array : DimArray

    Examples
    --------
    >>> import dimarray as da
    >>> a = da.DimArray([[[1,2,3]]])
    >>> a
    dimarray: 3 non-null elements (0 null)
    0 / x0 (1): 0 to 0
    1 / x1 (1): 0 to 0
    2 / x2 (3): 0 to 2
    array([[[1, 2, 3]]])
    >>> a.squeeze()
    dimarray: 3 non-null elements (0 null)
    0 / x2 (3): 0 to 2
    array([1, 2, 3])
    >>> a.squeeze(axis='x1')
    dimarray: 3 non-null elements (0 null)
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

    return self._constructor(res, newaxes, **self.attrs)

def _unflatten_dims(dims):
    """ ['a','b,c','d'] ==> ['a','b','c','d']
    """
    flatdims = []
    for d in dims:
        flatdims.extend(d.split(','))
    return flatdims


#
# Reshape by adding/removing as many singleton axes as needed to match prescribed dimensions
#
def reshape(self, *newdims, **kwargs):
    """ Add/remove/flatten dimensions to conform array to new dimensions
    
    Parameters
    ----------
    newdims : tuple or list or variable list of dimension names {str} 
        Any dimension now present in the array is added as singleton dimension
        Any dimension name containing a comma is interpreting as a flattening command.
        All dimensions to flatten have to exist already.

    transpose : bool
        if True, transpose dimensions to match new order (default True)
        otherwise, raise and Error if tranpose is needed (closer to original numpy's behaviour)

    Returns
    -------
    reshaped_array : DimArray 
        with reshaped_array.dims == tuple(newdims)

    See also
    --------
    flatten, unflatten, transpose, newaxis

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray([7,8])
    >>> a
    dimarray: 2 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    array([7, 8])

    >>> a.reshape(('x0','new'))
    dimarray: 2 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / new (1): None to None
    array([[7],
           [8]])

    >>> b = DimArray(np.arange(2*2*2).reshape(2,2,2))
    >>> b
    dimarray: 8 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (2): 0 to 1
    2 / x2 (2): 0 to 1
    array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[4, 5],
            [6, 7]]])

    >>> c = b.reshape('x0','x1,x2')
    >>> c
    dimarray: 8 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1,x2 (4): (0, 0) to (1, 1)
    array([[0, 1, 2, 3],
           [4, 5, 6, 7]])

    >>> c.reshape('x0,x1','x2')
    dimarray: 8 non-null elements (0 null)
    0 / x0,x1 (4): (0, 0) to (1, 1)
    1 / x2 (2): 0 to 1
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])
    """
    # parse input parameters
    # ...sequence or variable sequence
    if len(newdims) == 1 and type(newdims[0]) in (list, tuple):
        newdims = newdims[0]

    # ...tranpose parameter
    transpose = kwargs.pop('transpose', True)
    assert len(kwargs) == 0, "invalid parameters: {}".format(kwargs)

    # Do nothing if dimensions already match
    if tuple(newdims) == self.dims:
        return self

    # check that newd dimensions
    assert len(newdims) == len(set(newdims)), "must not contain duplicate axes !"

    # First unflatten the array to compare with flattened newdims
    o = self.unflatten()

    # Temporarily replace "," by ";" in any dimension with is NOT a flattened axis, and flatten all dimensions apart from that
    newdims_renamed = []
    for d in newdims:
        if ',' in d and d in o.dims:
            d = d.replace(',',';')
        newdims_renamed.append(d)
    newdims_unflattened = _unflatten_dims(newdims_renamed) 

    assert len(newdims_unflattened) == len(set(newdims_unflattened)), "must not contain duplicate axes !"

    for ax in o.axes:
        ax.name = ax.name.replace(',',';')

    # Remove unwanted singleton dimensions, if any
    for dim in o.dims:
        assert isinstance(dim, basestring), "newdims must be a tuple of axis names (`str`)"
        if dim not in newdims_unflattened:
            o = o.squeeze(dim)

    # Transpose array to match existing dimensions
    if transpose:
        o = o.transpose([dim for dim in newdims_unflattened if dim in o.dims])

    # check sortedness
    else:
        if tuple([d for d in o.dims if d in newdims_unflattened]) != tuple([d for d in newdims_unflattened if d in o.dims]):
            raise ValueError("First transpose the array before reshaping, or set parameter `transpose=True`")

    # Add missing dimensions by inserting singleton axes
    for i, dim in enumerate(newdims_unflattened):
        if dim not in o.dims:
            o = o.newaxis(dim, pos=i)

    # Group dimensions as required
    for i, d in enumerate(newdims_renamed):
        if ',' in d:
            o = o.flatten(d.split(','), insert=i)

    # Replace back ';' by ','
    for ax in o.axes:
        ax.name = ax.name.replace(';',',')

    if o.dims != tuple(newdims):
        raise ValueError("Could not perform reshaping, read documentation for acceptable arguments")

    return o


#
# Group/unflatten subsets of axes to perform operations on partly flattened array
#
def flatten(self, *dims, **kwargs):
    """Flatten all or a subset of dimensions

    Parameters
    ----------
    dims : list or tuple of axis names, optional
        by default, all dimensions
    reverse : bool, optional
        if True, reverse behaviour: dims are interpreted as 
        the dimensions to keep, and all the other dimensions are flattened
        default is False
    insert : int, optional
        position where to insert the flattened axis 
        (by default, any flattened dimension is inserted at 
        the position of the first axis involved in flattening)

    Returns
    -------
    flattened_array : DimArray
        appropriately reshaped, with collapsed dimensions as first axis (tuples)

    This is useful to do a regional mean with missing values

    Notes
    -----
    A tuple of axis names can be passed via the "axis" parameter of the transformation
    to trigger flattening prior to reducing an axis.

    See also
    --------
    reshape, transpose

    Examples
    --------

    Flatten all dimensions 

    >>> from dimarray import DimArray
    >>> a = DimArray([[1,2,3],[4,5,6]])
    >>> a
    dimarray: 6 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> b = a.flatten()
    >>> b
    dimarray: 6 non-null elements (0 null)
    0 / x0,x1 (6): (0, 0) to (1, 2)
    array([1, 2, 3, 4, 5, 6])

    >>> b.labels
    (array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], dtype=object),)

    Flatten a subset of dimensions only

    >>> from dimarray import DimArray
    >>> np.random.seed(0)
    >>> values = np.arange(2*3*4).reshape(2,3,4)
    >>> v = DimArray(values, axes=[('time', [1950,1955]), ('lat', np.linspace(-90,90,3)), ('lon', np.linspace(-180,180,4))])
    >>> v
    dimarray: 24 non-null elements (0 null)
    0 / time (2): 1950 to 1955
    1 / lat (3): -90.0 to 90.0
    2 / lon (4): -180.0 to 180.0
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])

    >>> w = v.flatten(('lat','lon'), insert=1)
    >>> w 
    dimarray: 24 non-null elements (0 null)
    0 / time (2): 1950 to 1955
    1 / lat,lon (12): (-90.0, -180.0) to (90.0, 180.0)
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])

    >>> np.all( w.unflatten() == v )
    True

    But be careful, the order matter !

    >>> v.flatten(('lon','lat'), insert=1)
    dimarray: 24 non-null elements (0 null)
    0 / time (2): 1950 to 1955
    1 / lon,lat (12): (-180.0, -90.0) to (180.0, 90.0)
    array([[ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11],
           [12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23]])

    Useful to average over a group of dimensions:

    >>> v.flatten(('lon','lat'), insert=0).mean(axis=0)
    dimarray: 2 non-null elements (0 null)
    0 / time (2): 1950 to 1955
    array([  5.5,  17.5])

    is equivalent to:

    >>> v.mean(axis=('lon','lat')) 
    dimarray: 2 non-null elements (0 null)
    0 / time (2): 1950 to 1955
    array([  5.5,  17.5])
    """
    reverse= kwargs.pop('reverse',False)
    #insert = kwargs.pop('insert', 0)
    insert = kwargs.pop('insert', None)
    assert len(kwargs) == 0, "invalid arguments: "+repr(kwargs.keys())

    reorder = False # internal flag to indicate the need to reorder dimensions

    # from variable list to tuple
    if len(dims) == 1 and type(dims[0]) in (list, tuple, set):
        dims = dims[0]
        if type(dims) is set:
            reorder = True

    if len(dims) == 0: 
        dims = self.dims

    if type(dims) not in (tuple, list, set):
        raise TypeError("dimensions to flatten must be a list or a tuple  or a set")

    # make sure dims contains axis names
    _ , dims = self._get_axes_info(dims)

    # reverse? mirror call
    assert type(reverse) is bool, "reverse must be a boolean !"
    if reverse:
        dims = [d for d in self.dims if d not in dims]

    # reorder dimensions to flatten and convert to tuple
    if reorder: # only if provided as set
        dims = [d for d in self.dims if d in dims] # do not reorder, this may matter
    dims = tuple(dims)

    n = len(dims) # number of dimensions to flatten
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

        return b.flatten(dims, insert=insert)

    # Create a new flattened axis
    newaxis = MultiAxis(*[ax for ax in self.axes if ax.name in dims])

    # New axes
    newaxes = [ax for ax in self.axes if ax.name not in dims]
    newaxes.insert(insert, newaxis)

    # Reshape the actual array values
    newshape = [ax.size for ax in newaxes]
    newvalues = self.values.reshape(newshape)

    # Define the new array
    new = self._constructor(newvalues, newaxes, **self.attrs)

    return new

group = deprecated_func(flatten, 'group')

def unflatten(self, axis=None):
    """ undo flatten (inflate array)

    Parameters
    ----------
    axis : int or str or None, optional
        axis to unflatten
        default to None to unflatten all

    Returns
    -------
    DimArray

    """
    # by default, unflatten all
    if axis is None:
        flattened_axes = [ax.name for ax in self.axes if isinstance(ax, MultiAxis)]
        obj = self
        for axis in flattened_axes:
            obj = obj.unflatten(axis=axis)
        return obj

    assert type(axis) in (str, int), "axis must be integer or string"

    group = self.axes[axis]    # axis to expand
    axis = self.dims.index(group.name) # make axis be an integer

    assert isinstance(group, MultiAxis), "can only unflatten a MultiAxis"

    newshape = self.shape[:axis] + tuple(ax.size for ax in group.axes) + self.shape[axis+1:]
    newvalues = self.values.reshape(newshape)
    newaxes = self.axes[:axis] + group.axes + self.axes[axis+1:]

    return self._constructor(newvalues, newaxes, **self.attrs)

ungroup = deprecated_func(unflatten, 'ungroup')

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

# def groupby(self, *dims):
#     """ group by one or several variables along which stat functions can be applied
#
#     .. note:: EXPERIMENTAL: will probably be removed or renamed in future releases
#
#     Parameters
#     ----------
#     *dims: variable list of dims to keep, all others are flattened
#
#     Returns
#     -------
#     GroupBy object
#
#     Notes 
#     -----
#     this method is experimental and may change in the future
#
#     Examples
#     --------
#     >>> from dimarray import DimArray
#     >>> a = DimArray([[1,2,3],[4,5,6]], [('x',[1,2]), ('items',['a','b','c'])])
#     >>> a = a.newaxis('y',4) # add an axis of length 4
#     >>> a.groupby('items').mean()
#     dimarray: 3 non-null elements (0 null)
#     0 / items (3): 'a' to 'c'
#     array([ 2.5,  3.5,  4.5])
#     """
#     obj = group(self, dims, reverse=True, insert=0).T
#     return GroupBy(obj, dims)
