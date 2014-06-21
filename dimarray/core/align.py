""" Functions associated to array alignment
"""
from collections import OrderedDict as odict
import numpy as np
import itertools
from axes import Axes, Axis
import warnings
from tools import is_DimArray

def broadcast_arrays(*arrays):
    """ Analogous to numpy.broadcast_arrays
    
    but with looser requirements on input shape
    and returns copy instead of views

    Parameters
    ----------
    arrays: variable list of DimArrays

    Returns
    -------
    list of DimArrays

    Examples
    --------
    Just as numpy's broadcast_arrays:

    >>> import dimarray as da
    >>> x = da.array([[1,2,3]])
    >>> y = da.array([[1],[2],[3]])
    >>> da.broadcast_arrays(x, y)
    [dimarray: 9 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (3): 0 to 2
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]]), dimarray: 9 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (3): 0 to 2
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])]
    """
    # give all objects the same dimension (without changing the size)
    arrays = align_dims(*arrays)

    # get axes object made of all non-singleton common axes
    try:
        axes = _get_axes(*arrays)

    # fails if axes are not aligned
    except AssertionError, msg:
        raise ValueError(msg)

    # now broadcast each DimArray along commmon axes
    newarrays = []
    for o in arrays:
        o = o.broadcast(axes)
        newarrays.append(o)

    return newarrays



def get_dims(*arrays):
    """ find all dimensions from a variable list of arrays
    """
    dims = []
    for o in arrays:
        for dim in o.dims:
            if dim not in dims:
                dims.append(dim)

    return dims


def _get_axes(*arrays):
    """ find list of axes from a list of axis-aligned DimArray objects
    """
    dims = get_dims(*arrays) # all dimensions present in objects
    axes = Axes()

    for dim in dims:

        common_axis = None

        for o in arrays:

            # skip missing dimensions
            if dim not in o.dims: continue

            axis = o.axes[dim]

            # update values
            if common_axis is None or (common_axis.size==1 and axis.size > 1):
                common_axis = axis

            # Test alignment for non-singleton axes
	    if not (axis.size == 1 or np.all(axis.values==common_axis.values)):
		raise ValueError("axes are not aligned")

        # append new axis
        axes.append(common_axis)


    return axes


def align_dims(*arrays):
    """ Align dimensions of a list of arrays so that they are ready for broadcast.
    
    Method: inserting singleton axes at the right place and transpose where needed.

    Examples
    --------
    >>> import dimarray as da
    >>> import numpy as np
    >>> x = da.DimArray(np.arange(2), dims=('x0',))
    >>> y = da.DimArray(np.arange(3), dims=('x1',))
    >>> da.align_dims(x, y)
    [dimarray: 2 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (1): None to None
    array([[0],
           [1]]), dimarray: 3 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (1): None to None
    1 / x1 (3): 0 to 2
    array([[0, 1, 2]])]
    """
    # If dimensions are already equal, do nothing
    lst = {o.dims for o in arrays}
    if len(lst) == 1:
        return arrays

    # Determine the dimensions of the result
    newdims = get_dims(*arrays) 

    # Reshape all DimArrays
    newarrays = []
    for o in arrays:
        o = o.reshape(newdims)
        newarrays.append(o)

    return newarrays


def align_axes(*arrays):
    """ align axes of a list of DimArray arrays by reindexing

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray([0,1,2],axes=[[0,1,2]])
    >>> b = DimArray([2,3],axes=[[2,3]])
    >>> align_axes(a, b)
    [dimarray: 3 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (4): 0 to 3
    array([  0.,   1.,   2.,  nan]), dimarray: 2 non-null elements (2 null)
    dimensions: 'x0'
    0 / x0 (4): 0 to 3
    array([ nan,  nan,   2.,   3.])]

    Also work on multi-dimensional arrays
     
    >>> a = DimArray([0,1], axes=[[0,1]]) # on 'x0' only
    >>> b = DimArray([[0,1],[2,3.],[4.,5.]], axes=[[0,1,2],[1,2]]) # one more element along the 1st dimension, 2nd dimension ignored
    >>> align_axes(a, b)
    [dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  0.,   1.,  nan]), dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (2): 1 to 2
    array([[ 0.,  1.],
           [ 2.,  3.],
           [ 4.,  5.]])]
    """
    # find the dimensiosn
    dims = get_dims(*arrays)

    arrays = list(arrays)
    for jj, d in enumerate(dims):

        # arrays which have that dimension
        ii = filter(lambda i: d in arrays[i].dims, range(len(arrays)))

        # common axis to reindex on
        ax_values = _common_axis(*[arrays[i].axes[d] for i in ii])

        # update arrays
        for i, o in enumerate(arrays):
            if i not in ii:
                continue
            if o.axes[d] == ax_values:
                continue

            arrays[i] = o.reindex_axis(ax_values, axis=d)

    return arrays


def _common_axis(*axes):
    """ find the common axis among a list of axes ==> proceed recursively
    """
    assert len(axes) > 0

    if len(axes) == 1:
        return axes[0]

    elif len(axes) == 2:
        ax0, ax1 = axes
        return ax0.union(ax1)

    else:
        ax0 = axes[0]
        ax1 = _common_axis(*axes[1:])
        return ax0.union(ax1)

def _check_stack_args(arrays, keys=None):
    """ generic function to deal with arguments for stacking
    accepts arrays as sequence or dict and returns 
    a list of keys and values
    """
    # convert dictionary to sequence + keys
    if isinstance(arrays, dict):
        if keys is None: keys = arrays.keys()
        arrays = arrays.values()
        
    # make sure the result is a sequence
    if type(arrays) not in (list, tuple):
	raise TypeError("argument must be a dictionary, list or tuple")

    # make sure keys exist
    if keys is None: keys = np.arange(len(arrays))

    return arrays, keys

def _check_stack_axis(axis, dims, default='unnamed'):
    """ check or get new axis name when stacking array or datasets
    (just to have that in one place)
    """
    if axis is None:
	axis = default
	if axis in dims:
	    i = 1
	    while default+"_{}".format(i) in dims:
		i+=1
	    axis = default+"_{}".format(i)

    if axis in dims:
	raise ValueError("please provide an axis name which does not \
		already exist")
    return axis

def stack(arrays, axis=None, keys=None, align=False):
    """ stack arrays along a new dimension (raise error if already existing)

    Parameters
    ----------
    arrays: sequence or dict of arrays
    axis, optional: str, new dimension along which to stack the array
    keys, optional: stack axis values, useful if array is a sequence, or a non-ordered dictionary
    align, optional: if True, align axes prior to stacking (Default to False)

    Returns
    -------
    DimArray: joint array

    See Also
    --------
    concatenate: join arrays along an existing dimension

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray([1,2,3])
    >>> b = DimArray([11,22,33])
    >>> stack([a, b], axis='stackdim', keys=['a','b'])
    dimarray: 6 non-null elements (0 null)
    dimensions: 'stackdim', 'x0'
    0 / stackdim (2): a to b
    1 / x0 (3): 0 to 2
    array([[ 1,  2,  3],
           [11, 22, 33]])
    """
    # make a sequence of arrays
    arrays, keys = _check_stack_args(arrays, keys)

    for a in arrays: 
        if not is_DimArray(a): raise TypeError('can only stack DimArray instances')

    # make sure the stacking dimension is OK (new)
    dims = get_dims(*arrays)
    axis = _check_stack_axis(axis, dims)

    # re-index axes if needed
    if align:
	arrays = align_axes(*arrays)

    # make it a numpy array
    data = [a.values for a in arrays]
    data = np.array(data)

    # new axis
    newaxis = Axis(keys, axis)

    # find common axes
    try: 
	axes = _get_axes(*arrays)
    except ValueError, msg: 
	if 'axes are not aligned' in repr(msg):
	    msg = 'axes are not aligned\n ==> Try passing `align=True`' 
	raise ValueError(msg)

    # new axes
    #newaxes = axes[:pos] + [newaxis] + axes[pos:] 
    newaxes = [newaxis] + axes

    # create dimarray
    _constructor = arrays[0]._constructor # DimArray
    return _constructor(data, axes=newaxes)


def concatenate(arrays, axis=0, check_other_axes=True):
    """ concatenate several DimArrays

    Parameters
    -----------
    arrays: list of DimArrays
    axis  : axis along which to concatenate

    Returns
    -------
    concatenated DimArray 

    See Also
    --------
    stack: join arrays along a new dimension

    Examples
    --------

    1-D

    >>> from dimarray import DimArray
    >>> a = DimArray([1,2,3], axes=[['a','b','c']])
    >>> b = DimArray([4,5,6], axes=[['d','e','f']])
    >>> concatenate((a, b))
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (6): a to f
    array([1, 2, 3, 4, 5, 6])

    2-D

    >>> a = DimArray([[1,2,3],[11,22,33]])
    >>> b = DimArray([[4,5,6],[44,55,66]])
    >>> concatenate((a, b), axis=0)
    dimarray: 12 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (4): 0 to 1
    1 / x1 (3): 0 to 2
    array([[ 1,  2,  3],
           [11, 22, 33],
           [ 4,  5,  6],
           [44, 55, 66]])
    >>> concatenate((a, b), axis='x1')
    dimarray: 12 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (6): 0 to 2
    array([[ 1,  2,  3,  4,  5,  6],
           [11, 22, 33, 44, 55, 66]])
    """
    assert type(arrays) in (list, tuple), "arrays must be list or tuple, got {}:{}".format(type(arrays), arrays)

    if type(axis) is not int:
        axis = arrays[0].dims.index(axis)

    values = np.concatenate([a.values for a in arrays], axis=axis)

    _get_subaxes = lambda x: [ax for i, ax in enumerate(a.axes) if i != axis]
    subaxes = _get_subaxes(arrays[0])
    #assert np.all(_get_subaxes(a) == subaxes for a in arrays),"some axes do not match"

    # concatenate axis values
    #newaxisvalues = np.concatenate([a.axes[axis].values for a in arrays])
    #newaxis = Axis(newaxisvalues, name=arrays[0].dims[axis])
    newaxis = concatenate_axes([a.axes[axis] for a in arrays])

    # check that other axes match
    if check_other_axes:
        for i,a in enumerate(arrays[1:]):
            if not _get_subaxes(a) == subaxes:
                msg = "First array:\n{}\n".format(subaxes)
                msg += "{}th array:\n{}\n".format(i,_get_subaxes(a))
                raise ValueError("other axes to not match. Check out `aggregate` method")

    newaxes = subaxes[:axis] + [newaxis] + subaxes[axis:]

    return arrays[0]._constructor(values, newaxes)

#def _aggregate_axes(arrays):
#    """ build a common Axes object from a list of arrays
#    """
#    for ax
def concatenate_axes(axes):
    """ concatenate Axis objects

    axes: list of Axis objects

    >>> a = Axis([1,2,3],'x0')
    >>> b = Axis([5,6,7],'x0')
    >>> ax = concatenate_axes((a, b))
    >>> ax.name
    'x0'
    >>> ax.values
    array([1, 2, 3, 5, 6, 7])
    """
    #assert np.iterable(axes) and axes
    #if not isinstance(axes[0], Axis): raise TypeError()
    if len({ax.name for ax in axes}) != 1: 
        print axes
        raise ValueError("axis names differ!")
    values = np.concatenate([ax.values for ax in axes])
    return Axis(values, axes[0].name)

def aggregate(arrays, check_overlap=True):
    """ like a multi-dimensional concatenate

    Parameters
    ----------
        arrays: sequence of DimArrays

        check_overlap, optional: if True, check that arrays do not overlap (to avoid data loss)
            If any two elements overlap, keep the one which is not NaN, if applicable
            or raise an error if two valid values overlap

            Default is True to reduce the risk of errors, but this makes the operation
            less performant since every time a copy of the subarray is extracted 
            and tested for NaNs. Consider setting check_overlap to False for large
            arrays for a well-tested problems, if the valid-nan selection is not 
            required.

    Notes
    -----
        Probably a bad idea to have duplicate axis values (not tested)

    TODO: add support for missing values other than np.nan

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray([[1.,2,3]],axes=[('line',[0]), ('col',['a','b','c'])])
    >>> b = DimArray([[4],[5]], axes=[('line',[1,2]), ('col',['d'])])
    >>> c = DimArray([[22]], axes=[('line',[2]), ('col',['b'])])
    >>> d = DimArray([-99], axes=[('line',[4])])
    >>> aggregate((a,b,c,d))
    dimarray: 10 non-null elements (6 null)
    dimensions: 'line', 'col'
    0 / line (4): 0 to 4
    1 / col (4): a to d
    array([[  1.,   2.,   3.,  nan],
           [ nan,  nan,  nan,   4.],
           [ nan,  22.,  nan,   5.],
           [-99., -99., -99., -99.]])

    But beware of overlapping arrays. The following will raise an error:

    >>> a = DimArray([[1.,2,3]],axes=[('line',[0]), ('col',['a','b','c'])])
    >>> b = DimArray([[4],[5]], axes=[('line',[0,1]), ('col',['b'])])
    >>> try:
    ...            aggregate((a,b))    
    ... except ValueError, msg:
    ...            print msg
    Overlapping arrays: set check_overlap to False to suppress this error.

    Can set check_overlap to False to let it happen anyway (the latter array wins)

    >>> aggregate((a,b), check_overlap=False)  
    dimarray: 4 non-null elements (2 null)
    dimensions: 'line', 'col'
    0 / line (2): 0 to 1
    1 / col (3): a to c
    array([[  1.,   4.,   3.],
           [ nan,   5.,  nan]])

    Note that if NaNs are present on overlapping, the valid data are kept

    >>> a = DimArray([[1.,2,3]],axes=[('line',[1]), ('col',['a','b','c'])])
    >>> b = DimArray([[np.nan],[5]], axes=[('line',[1,2]), ('col',['b'])])
    >>> aggregate((a,b)) # does not overwrite `2` at location (1, 'b')
    dimarray: 4 non-null elements (2 null)
    dimensions: 'line', 'col'
    0 / line (2): 1 to 2
    1 / col (3): a to c
    array([[  1.,   2.,   3.],
           [ nan,   5.,  nan]])
    """
    # list of common dimensions
    dims = get_dims(*arrays)

    # build a common Axes object 
    axes = Axes()
    for d in dims:
        newaxis = concatenate_axes([a.axes[d] for a in arrays if d in a.dims])
        newaxis.values = np.unique(newaxis.values) # unique values
        axes.append(newaxis)

    # Fill in an array
    newarray = arrays[0]._constructor(None, axes=axes, dtype=arrays[0].dtype)
    for a in arrays:

        indices = {ax.name:ax.values for ax in a.axes}

        if check_overlap:

            # look for nans in replaced and replacing arrays
            subarray = newarray.take(indices, broadcast_arrays=False).values
            subarray_is_nan = np.isnan(subarray)
            newvalues_is_nan = np.isnan(a.values)

            # check overlapping
            overlap_values  = ~subarray_is_nan & ~newvalues_is_nan
            if np.any(overlap_values):
                raise ValueError("Overlapping arrays: set check_overlap to False to suppress this error.")

            # only take new non-nan values
            newvalues = np.where(newvalues_is_nan, subarray, a.values) 

        else:
            newvalues = a.values

        # The actual operation is done by put
        newarray.put(newvalues, indices=indices, inplace=True, convert=True, broadcast_arrays=False)

    # That's it !

    return newarray


##
## Kind of all-purpose align method: not very relevant so it may be removed at some point
## it is also used by da.array
##
#def join(data, keys=None, axis=None, cls=None):
#    """ initialize a DimArray from a dictionary of smaller dimensional DimArray
#
#    ==> align and stack a sequence of dimarrays
#
#    Convenience method for: Dataset(data, keys).to_array(axis)
#
#    input:
#        - data : list or dict of DimArrays
#        - keys, optional : labels of the first dimension (if dict, only useful for ordering)
#        - axis, optional : dimension name along which to aggregate data (default "unnamed")
#
#    output:
#        - new DimArray object, with axis alignment (reindexing)
#
#    See Also:
#    ---------
#    array, stack, concatenate, Dataset
#    """
#    from dimarray.dataset import Dataset, odict # 
#    from dimarray import DimArray
#    #data = _get_list_arrays(data, keys)        
#
#    if not isinstance(data, dict):
#        assert isinstance(data, list), "DimArray.from_arrays only acceps dict and list, got {}: {}".format(type(data), data)
#        if keys is None:
#            keys = []
#            for i, v in enumerate(data):
#                assert isinstance(v, DimArray), "DimArray.from_arrays only acceps dict and list of DimArray objects, got {}: {}".format(type(v), v)
#                if not hasattr(v, "name") or v.name is None:
#                    name = i
#
#                else:
#                    name = v.name
#                keys.append(name)
#        data = {keys[i]:v for i, v in enumerate(data)}
#    
#    return _join_from_dict(data, keys=keys, axis=axis, cls=cls)
#
#def _join_from_dict(dict_, keys=None, axis=None, cls=None):
#    """ Initialize a DimArray for a dictionary of DimArrays
#
#    keys, optional: re-order the keys 
#    axis, optional: give a name to the keys axis
#    """
#    assert isinstance(dict_, dict)
#    from dimarray.dataset import Dataset
#    if keys is None: keys = dict_.keys()
#    data = Dataset(dict_)
#    if cls is None: 
#        _constructor = None
#    else:
#        _constructor = cls._constructor
#    return data.to_array(axis=axis, keys=keys, _constructor=_constructor)

#def aligned(objects, skip_missing=True, skip_singleton=True):
#    """ test whether common non-singleton axes are equal
#    """
#    # check whether all objects have the same dimensions
#    if not skip_missing:
#        set_of_dims = {o.dims for o in objects}
#        if len(set_of_dims) > 1:
#            return False
#
#    # test whether common non-singleton axes are equal
#    try: 
#        axes = _get_axes(*objects)
#    except:
#        return False
#
#    # test whether all existing dimensions have same size
#    if not skip_singleton:
#        dims = [ax.name for ax in axes]
#        for dim in dims:
#            set_of_sizes = {o.axes[dim].size for o in objects}
#            if len(set_of_sizes) > 1:
#                return False
#
#    return True
