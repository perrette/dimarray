""" Functions and dimarray methods associated to array alignment
"""
from collections import OrderedDict as odict
import itertools
import warnings
import numpy as np

from dimarray.config import get_option
from dimarray.tools import is_DimArray
from dimarray.core.axes import Axes, Axis
from dimarray.core.indexing import locate_many

__all__ = ["broadcast_arrays", "align", "stack", "concatenate"]

def get_dims(*arrays):
    """ find all dimensions from a variable list of arrays (or any object with `axes` attribute)
    Note: not in public API, but used by other modules
    """
    dims = []
    for o in arrays:
        for ax in o.axes:
            if ax.name not in dims:
                dims.append(ax.name)
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
    Note : not part of public API, but used in other dimarray modules

    Examples
    --------
    >>> import dimarray as da
    >>> import numpy as np
    >>> x = da.DimArray(np.arange(2), dims=('x0',))
    >>> y = da.DimArray(np.arange(3), dims=('x1',))
    >>> align_dims(x, y)
    [dimarray: 2 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (1): None to None
    array([[0],
           [1]]), dimarray: 3 non-null elements (0 null)
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

def broadcast_arrays(*arrays):
    """ Analogous to numpy.broadcast_arrays
    
    but with looser requirements on input shape
    and returns copy instead of views

    Parameters
    ----------
    arrays : variable list of DimArrays

    Returns
    -------
    list of DimArrays

    Examples
    --------
    Just as numpy's broadcast_arrays

    >>> import dimarray as da
    >>> x = da.DimArray([[1,2,3]])
    >>> y = da.DimArray([[1],[2],[3]])
    >>> da.broadcast_arrays(x, y)
    [dimarray: 9 non-null elements (0 null)
    0 / x0 (3): 0 to 2
    1 / x1 (3): 0 to 2
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]]), dimarray: 9 non-null elements (0 null)
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

def _common_axis(axes, join):
    """ find the common axis among a list of axes ==> proceed recursively
    """
    assert len(axes) > 0

    # recursion end
    if len(axes) == 1:
        return axes[0]

    # recursive call
    ax0 = axes[0]
    ax1 = _common_axis(axes[1:],join)

    # special cases
    # do not include None unless we have a singleton
    if ax0[0] is None:
        return ax1
    if len(ax1) == 1 and ax1[0] is None:
        return ax0

    # TODO: make a separate version of the axis module
    # with helper functions that do the basic work, without the whole
    # Axis machinery. This may limit the possibility of optimizing the 
    # Axes via hidden attributes, but this would also make things simpler
    # and prevents false "good" ideas (such as indeed, adding hidden attributes)
    if join == 'outer':
        com_axis = ax0.union(ax1)
    else:
        com_axis = ax0.intersection(ax1)
    return com_axis

def _get_aligned_axes(arrays, join='outer', axis=None , sort=False, strict=False):
    """From a list of arrays, or any object with `axes` attributes, 
    a new list of axes.
    """
    # find the dimensions
    if axis is None:
        dims = get_dims(*arrays)
    elif isinstance(axis, basestring):
        dims = [axis]
    else:
        if not isinstance(axis, basestring):
            raise ValueError("align: axis must be provided as a string")

    axes = Axes()

    for jj, d in enumerate(dims):

        # arrays which have that dimension
        ii = filter(lambda i: d in arrays[i].dims, range(len(arrays)))

        if strict and len(ii) != len(arrays):
            raise ValueError("align (strict=True): some arrays lack dimension {}".format(d))

        # common axis to reindex on
        ax = _common_axis([arrays[i].axes[d] for i in ii], join)

        if sort:
            ax.sort()

        axes.append(ax)

    # assert len(axes) > 0

    return axes

def align(arrays, join='outer', axis=None , sort=False, strict=False):
    """Align axes of a list of DimArray arrays by reindexing

    Parameters
    ----------
    array1, array2, ... : variable list of DimArrays or Datasets
    join : {"outer", "inner"}, optional
        method to find the common axis
        "outer" : union of all axes, missing values filled with NaNs
        "inner" : intersection of all axes
        Default to "outer" (can be changed with `dimarray.set_option('align.join","inner")`)
    sort : bool, optional
        Sort the axis prior to aligning.
        default to False
    axis : str, optional
        default to None : align all axes
        (must be a string since the axes do not necessarily match)
    strict : bool, optional
        if True, check that all arrays have the same dimensions

    Returns
    -------
    aligned_array1, aligned_array2, ... : list of aligned DimArrays (or Dataset)

    See Also
    --------
    `DimArray.reindex_axis`, `DimArray.reindex_like`

    Examples
    --------
    >>> from dimarray import DimArray, align
    >>> a = DimArray([0,1,2],axes=[[0,1,2]])
    >>> b = DimArray([1,2,3],axes=[[1,2,3]])
    >>> align([a, b])
    [dimarray: 3 non-null elements (1 null)
    0 / x0 (4): 0 to 3
    array([  0.,   1.,   2.,  nan]), dimarray: 3 non-null elements (1 null)
    0 / x0 (4): 0 to 3
    array([ nan,   1.,   2.,   3.])]
    >>> align([a, b], join='inner')
    [dimarray: 2 non-null elements (0 null)
    0 / x0 (2): 1 to 2
    array([1, 2]), dimarray: 2 non-null elements (0 null)
    0 / x0 (2): 1 to 2
    array([1, 2])]

    Also work on multi-dimensional arrays
     
    >>> a = DimArray([0,1], axes=[[0,1]]) # on 'x0' only
    >>> b = DimArray([[0,1],[2,3.],[4.,5.]], axes=[[0,1,2],[1,2]]) # one more element along the 1st dimension, 2nd dimension ignored
    >>> align([a, b])
    [dimarray: 2 non-null elements (1 null)
    0 / x0 (3): 0 to 2
    array([  0.,   1.,  nan]), dimarray: 6 non-null elements (0 null)
    0 / x0 (3): 0 to 2
    1 / x1 (2): 1 to 2
    array([[ 0.,  1.],
           [ 2.,  3.],
           [ 4.,  5.]])]
    """
    # join = kwargs.pop('join', get_option('align.join'))
    # sort = kwargs.pop('sort', False)
    # axis = kwargs.pop('axis', None)
    # strict = kwargs.pop('strict', False)
    # if len(kwargs) > 0:
    #     raise TypeError("align() got unexpected argument(s): "+", ".join(kwargs.keys()))
    if not (isinstance(arrays, list) or isinstance(arrays, tuple)):
        raise ValueError("align: only accepts list or tuple arguments. Got: {}".format(type(arrays)))

    # convert any scalar to dimarray
    from dimarray import DimArray, Dataset
    arrays = [a for a in arrays] # convert to list
    for i, a in enumerate(arrays):
        if not isinstance(a, DimArray) and not isinstance(a, Dataset):
            if np.isscalar(a):
                arrays[i] = DimArray(a)
            else:
                raise TypeError("can only align DimArray and Dataset instances, got: {}".format(type(a)))

    # find the common axes
    axes = _get_aligned_axes(arrays, axis=axis, join=join, sort=sort, strict=strict)

    # update arrays
    for ax in axes:
        for i, o in enumerate(arrays):
            if ax.name not in o.dims: 
                continue
            if np.all(o.axes[ax.name] == ax):
                continue
            arrays[i] = o.reindex_axis(ax)

    return arrays

align_ = align  # for internal use, so that it does not conflict with "align" parameter

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

    if type(axis) is int:
        raise TypeError("axis must be a str (new axis name)")

    if axis in dims:
        raise ValueError("please provide an axis name which does not \
                already exist, or use `concatenate`")
    return axis

def stack(arrays, axis=None, keys=None, align=False, **kwargs):
    """ stack arrays along a new dimension (raise error if already existing)

    Parameters
    ----------
    arrays : sequence or dict of arrays
    axis : str, optional
        new dimension along which to stack the array
    keys : array-like, optional
        stack axis values, useful if array is a sequence, or a non-ordered dictionary
    align : bool, optional
        if True, align axes prior to stacking (Default to False)
    **kwargs : optional key-word arguments passed to align, if align is True

    Returns
    -------
    DimArray : joint array

    See Also
    --------
    concatenate : join arrays along an existing dimension
    swapaxes : to modify the position of the newly inserted axis

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray([1,2,3])
    >>> b = DimArray([11,22,33])
    >>> stack([a, b], axis='stackdim', keys=['a','b'])
    dimarray: 6 non-null elements (0 null)
    0 / stackdim (2): 'a' to 'b'
    1 / x0 (3): 0 to 2
    array([[ 1,  2,  3],
           [11, 22, 33]])
    """
    assert not isinstance(axis, int), "axis must be a str (you are creating a new axis)"

    # make a sequence of arrays
    arrays, keys = _check_stack_args(arrays, keys)

    for a in arrays: 
        if not is_DimArray(a): raise TypeError('can only stack DimArray instances')

    # make sure the stacking dimension is OK (new)
    dims = get_dims(*arrays)
    axis = _check_stack_axis(axis, dims)

    # re-index axes if needed
    if align:
        kwargs['strict'] = True
        arrays = align_(arrays, **kwargs)

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

def _concatenate_axes(axes):
    """ concatenate Axis objects

    axes: list of Axis objects

    >>> a = Axis([1,2,3],'x0')
    >>> b = Axis([5,6,7],'x0')
    >>> ax = _concatenate_axes((a, b))
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

def concatenate(arrays, axis=0, _no_check=False, align=False, **kwargs):
    """ concatenate several DimArrays

    Parameters
    -----------
    arrays : list of DimArrays
        arrays to concatenate
    axis : int or str 
        axis along which to concatenate (must exist)
    align : bool, optional
        align secondary axes before joining on the primary
        axis `axis`. Default to False.
    **kwargs : optional key-word arguments passed to align, if align is True

    Returns
    -------
    concatenated DimArray 

    See Also
    --------
    stack: join arrays along a new dimension
    align: align arrays

    Examples
    --------

    1-D

    >>> from dimarray import DimArray
    >>> a = DimArray([1,2,3], axes=[['a','b','c']])
    >>> b = DimArray([4,5,6], axes=[['d','e','f']])
    >>> concatenate((a, b))
    dimarray: 6 non-null elements (0 null)
    0 / x0 (6): 'a' to 'f'
    array([1, 2, 3, 4, 5, 6])

    2-D

    >>> a = DimArray([[1,2,3],[11,22,33]])
    >>> b = DimArray([[4,5,6],[44,55,66]])
    >>> concatenate((a, b), axis=0)
    dimarray: 12 non-null elements (0 null)
    0 / x0 (4): 0 to 1
    1 / x1 (3): 0 to 2
    array([[ 1,  2,  3],
           [11, 22, 33],
           [ 4,  5,  6],
           [44, 55, 66]])
    >>> concatenate((a, b), axis='x1')
    dimarray: 12 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (6): 0 to 2
    array([[ 1,  2,  3,  4,  5,  6],
           [11, 22, 33, 44, 55, 66]])
    """
    # input argument check
    if not type(arrays) in (list, tuple):
        raise ValueError("arrays must be list or tuple, got {}:{}".format(type(arrays), arrays))
    arrays = [a for a in arrays]

    from dimarray import DimArray, Dataset

    for i, a in enumerate(arrays):
        if isinstance(a, Dataset):
            msg = "\n==>Note: you may use `concatenate_ds` for Datasets"
            raise ValueError("concatenate: expected DimArray. Got {}".format(type(a))+msg)
        elif np.isscalar(a):
            arrays[i] = DimArray(a)
        if not isinstance(a, DimArray):
            raise ValueError("concatenate: expected DimArray. Got {}".format(type(a)))

    if type(axis) is not int:
        axis = arrays[0].dims.index(axis)
    dim = arrays[0].dims[axis]

    # align secondary axes prior to concatenate
    # TODO: just encourage user to use align outside this function
    # and remove argument passing
    if align:
        kwargs['strict'] = True
        for ax in arrays[0].axes:
            if ax.name != dim:
                arrays = align_(arrays, axis=ax.name, **kwargs)

    values = np.concatenate([a.values for a in arrays], axis=axis)

    _get_subaxes = lambda x: [ax for i, ax in enumerate(arrays[0].axes) if i != axis]
    subaxes = _get_subaxes(arrays[0])

    # concatenate axis values
    newaxis = _concatenate_axes([a.axes[axis] for a in arrays])

    if not align and not _no_check:
        # check that other axes match
        for ax in subaxes:
            for a in arrays:
                if not np.all(a.axes[ax.name].values == ax.values):
                    raise ValueError("contatenate: secondary axes do not match. Align first? (`align=True`)")
        # print arrays[0]
        # for i,a in enumerate(arrays[1:]):
        #     if not _get_subaxes(a) == subaxes:
        #         msg = "First array:\n{}\n".format(subaxes)
        #         msg += "{}th array:\n{}\n".format(i,_get_subaxes(a))
        #         raise ValueError("contatenate: secondary axes do not match. Align first? (`align=True`)")
        #     print a
        # print '==> arrays look ok'

    newaxes = subaxes[:axis] + [newaxis] + subaxes[axis:]

    return arrays[0]._constructor(values, newaxes)

##############################################################
# The functions below are meant to be used as DimArray methods
##############################################################

#
# Reindex axis
#
def reindex_axis(self, values, axis=0, fill_value=np.nan, raise_error=False, method=None):
    """ reindex an array along an axis

    Parameters
    ----------
    values : array-like or Axis
        new axis values
    axis : int or str, optional
        axis number or name
    fill_value: bool, optional
        Fill data to use for missing axis value, 
        if `raise_error` is False.
    raise_error : bool, optional
        if True, raise error when an axis value is not present 
        otherwise just replace with `fill_value`. Defaulf is False
    method : {None, 'left', 'right'}
        method to fill the gaps (default None)
        If 'left' or 'right', just pass along to numpy.searchsorted.

    Returns
    -------
    dimarray: DimArray instance

    Examples
    --------
    Basic reindexing: fill missing values with NaN

    >>> import dimarray as da
    >>> a = da.DimArray([1,2,3],axes=[('x0', [1,2,3])])
    >>> b = da.DimArray([3,4],axes=[('x0',[1,3])])
    >>> b.reindex_axis([1,2,3])
    dimarray: 2 non-null elements (1 null)
    0 / x0 (3): 1 to 3
    array([  3.,  nan,   4.])

    Or replace with anything else, like -9999

    >>> b.reindex_axis([1,2,3], fill_value=-9999)
    dimarray: 3 non-null elements (0 null)
    0 / x0 (3): 1 to 3
    array([    3, -9999,     4])
    """
    if isinstance(values, Axis):
        newaxis = values
        values = newaxis.values
        axis = newaxis.name
    elif np.isscalar(values) or type(values) is slice:
        raise TypeError("Please provide list, array-like or Axis object to perform re-indexing")
    else:
        values = np.asarray(values)

    # Get indices
    ax = self.axes[axis]
    # indices = ax.loc(values, mode='clip', side=method)
    indices = locate_many(ax.values, values, side=method or 'left')
    newobj = self.take_axis(indices, axis, indexing='position')

    # Replace mismatch with missing values?
    mask = ax.values.take(indices) != values
    if np.any(mask):
        if raise_error:
            raise IndexError("Some values where not found in the axis: {}".format(values[mask]))
        if method is None:
            newobj.put(mask, fill_value, axis=axis, inplace=True, indexing="position", cast=True)
        # Make sure the axis values match the requested new axis
        newobj.axes[axis][mask] = values[mask]

    return newobj

def reindex_axis_with_pandas(obj, values, axis=0, fill_value=np.nan):
    """ Convert to and from pandas to use a faster (?) indexing method
    """

    import pandas
    pandasobj = obj.to_pandas()

    axis_id, axis_nm = obj._get_axis_info(axis)

    try:
        newpandas = pandasobj.reindex_axis(values, axis=axis_id, fill_value=fill_value)
    except TypeError:
        # older versions of pandas do not have the fill_value parameter
        newpandas = pandasobj.reindex_axis(values, axis=axis_id)

    newobj = obj.from_pandas(newpandas) # use class method from_pandas
    newobj.attrs.update(obj.attrs)    # add metadata back
    newobj.axes[axis_id].name = axis_nm  # give back original name

    return newobj


def reindex_like(self, other, **kwargs):
    """ reindex_like : re-index like another dimarray / axes instance

    Applies reindex_axis on each axis to match another DimArray

    Parameters
    ----------
    other : DimArray or Axes instance
    **kwargs : 

    Returns
    -------
    DimArray

    Notes
    -----
    only reindex axes which are present in other

    Examples
    --------
    >>> import dimarray as da
    >>> b = da.DimArray([3,4],('x0',[1,3]))
    >>> c = da.DimArray([[1,2,3], [1,2,3]],[('x1',["a","b"]),('x0',[1, 2, 3])])
    >>> b.reindex_like(c)
    dimarray: 2 non-null elements (1 null)
    0 / x0 (3): 1 to 3
    array([  3.,  nan,   4.])
    """
    if hasattr(other, 'axes'):
        axes = other.axes
    elif isinstance(other, Axes):
        axes = other
    else:
        raise TypeError('expected DimArray or Axes, got {}: {}'.format(type(other), other))

    newdims = [ax2.name for ax2 in axes]
    obj = self
    for ax in self.axes:
        if ax.name in newdims:
            newaxis = axes[ax.name].values
            obj = obj.reindex_axis(newaxis, axis=ax.name, **kwargs)

    return obj


def sort_axis(a, axis=0, key=None, kind='quicksort'):
    """ sort an axis 

    Parameters
    ----------
    a : DimArray (this argument is pre-assigned when using as bound method)
    axis : int or str, optional
        axis by position (int) or name (str) (default: 0)
    key : callable or dict-like, optional
        function that is called on each axis label and 
        whose return value is used for sorting instead of axis label.
        Any other object with __getitem__ attribute may also be used as key,
        such as a dictionary.
        If None (the default), axis label is used for sorting.
    kind : str, optional
        sort algorigthm (see numpy.sort for more info)

    Returns
    --------
    sorted : new DimArray with sorted axis

    Examples
    --------
    Basic

    >>> from dimarray import DimArray
    >>> a = DimArray([10,20,30], labels=[2, 0, 1])
    >>> a
    dimarray: 3 non-null elements (0 null)
    0 / x0 (3): 2 to 1
    array([10, 20, 30])

    >>> a.sort_axis()
    dimarray: 3 non-null elements (0 null)
    0 / x0 (3): 0 to 2
    array([20, 30, 10])

    >>> a.sort_axis(key=lambda x: -x)
    dimarray: 3 non-null elements (0 null)
    0 / x0 (3): 2 to 0
    array([10, 30, 20])

    Multi-dimensional
     
    >>> a = DimArray([[10,20,30],[40,50,60]], labels=[[0, 1], ['a','c','b']])
    >>> a.sort_axis(axis=1)
    dimarray: 6 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (3): 'a' to 'c'
    array([[10, 30, 20],
           [40, 60, 50]])
    """
    index = a.axes[axis].values

    # convert key to a function
    if key is None:
        ii = index.argsort(kind=kind) # the default
    else:
        if not hasattr(key, '__call__') and hasattr(key, '__getitem__'):
            key = key.__getitem__
        ii = argsort(index, key)

    return a.take_axis(ii, axis=axis, indexing='position')


def argsort(seq, key=None):
    """ equivalent of numpy's argsort in basic python

    Modified after http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

    >>> a = ['a', 'd', 'c']
    >>> argsort(a)
    [0, 2, 1]
    >>> argsort(a, key=lambda x: {'a':2,'c':1,'d':0}[x])
    [1, 2, 0]
    """
    if key is None:
        _key = seq.__getitem__
    else:
        _key = lambda x: key(seq.__getitem__(x))
    return sorted(range(len(seq)), key=_key)
