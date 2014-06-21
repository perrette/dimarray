""" Module to deal with indexing
"""
import numpy as np
import functools
import copy 

from axes import Axis, Axes, GroupedAxis, is_regular, make_multiindex
from tools import is_DimArray
from dimarray.config import get_option

__all__ = ["take", "put", "reindex_axis", "reindex_like"]

#TOLERANCE=1e-8
TOLERANCE=None

def is_boolean_index(indices, shape):
    """ check if something like a[a>2] is performed for ndim > 1
    """
    #indices = np.index_exp[indices]
    #if len(shape) > 1 and len(indices) == 1:
    if isinstance(indices, np.ndarray) or is_DimArray(indices):
        if indices.shape == shape:
            if indices.dtype == np.dtype(bool):
                return True

    return False # otherwise

def _fill_ellipsis(indices, ndim):
    """ replace Ellipsis
    """
    if indices is Ellipsis:
        return slice(None)

    elif type(indices) is not tuple:
        return indices

    # Implement some basic patterns
    is_ellipsis = np.array([ix is Ellipsis for ix in indices])
    ne = is_ellipsis.sum()

    if ne == 0:
        return indices

    elif ne > 1:
        raise NotImplementedError('Only support one Ellipsis')

    # From here, one Ellipsis is present
    i = np.where(is_ellipsis)[0]
    indices = indices[:i] + (slice(None),)*(ndim - len(indices) + 1) + indices[(i+1):]
    return indices

def array_indices(indices_numpy, shape):
    """ Replace non-iterable (incl. slices) with arrays

    indices_numpy: multi-index
    shape: shape of array to index
    """
    dummy_ix = []
    for i,ix in enumerate(indices_numpy):
        if not np.iterable(ix):
            if type(ix) is slice:
                ix = np.arange(shape[i])[ix]
            else:
                ix = np.array([ix])
        
        # else make sure we have an array, and if book, evaluate to arrays
        else:
            ix = np.asarray(ix)
            if ix.dtype is np.dtype(bool):
                ix = np.where(ix)[0]

        dummy_ix.append(ix)
    return dummy_ix

def broadcast_indices(indices):
    """ if any array index is present, broadcast all arrays and integer indices on the same shape
    """
    aindices = []

    # convert all booleans, and scan the indices to get the size
    size = None
    for i,ix in enumerate(indices):

        if np.iterable(ix) and np.asarray(ix).dtype is np.dtype(bool):
            ix = np.where(ix)[0]

        if np.iterable(ix):

            if size is None: 
                size = np.size(ix)

            # consistency check
            elif size != np.size(ix):
                print size, np.size(ix)
                raise ValueError("array-indices could not be broadcast on the same shape (got {} and {}, try box[...] or take(..., broadcast_arrays=False) if you intend to sample values along several dimensions independently)".format(size, np.size(ix)))

        aindices.append(ix)

    # Now convert all integers to the same size, if applicable
    if size is not None:
        for i,ix in enumerate(aindices):
            if not np.iterable(ix) and not type(ix) is slice:
                aindices[i] = np.zeros(size, dtype=type(ix)) + ix

    return aindices

def ix_(indices_numpy, shape):
    # convert to matlab-like compatible indices
    """ convert numpy-like to matlab-like indices

    indices_numpy: indices to convert
    shape: shape of the array, to convert slices
    """
    dummy_ix = array_indices(indices_numpy, shape)
    return np.ix_(*dummy_ix)

#def _remove_trailing_slice(indices_numpy):
#    """ remove trailing slice(None) from a tuple index
#    """
    # matching array size is useful 

def _get_keyword_indices(obj, indices, axis=0):
    """ get dictionary of indices
    """
    try:
        axes = obj.axes
    except:
        raise TypeError(" must be a DimArray instance !")

    # convert to dictionary
    if type(indices) is tuple or isinstance(indices, dict) or isinstance(indices, Axis):
        assert axis in (None, 0), "cannot have axis > 0 for tuple (multi-dimensional) indexing"

    if isinstance(indices, Axis):
        newaxis = indices
        kw = {newaxis.name: newaxis.values}

    elif type(indices) is tuple:
        kw = {axes[i].name: ind for i, ind in enumerate(indices)}

    elif type(indices) in (list, np.ndarray) or np.isscalar(indices):
        kw = {axes[axis].name:indices}

    elif isinstance(indices, dict):
        kw = indices

    else:
        raise TypeError("indices not understood: {}, axis={}".format(indices, axis))

    return kw

@property
def size(self):
    return np.size(self.values)


def take(obj, indices, axis=0, indexing="values", tol=TOLERANCE, keepdims=False, broadcast_arrays=True, mode='raise'):
    """ Retrieve values from a DimArray

    Parameters
    ----------

        - self or obj: DimArray (ignore this parameter if accessed as bound method)
        - indices  : int or list or slice (single-dimensional indices)
                     or a tuple of those (multi-dimensional)
                     or `dict` (`axis name` : `indices`)
        - axis     : int or str
        - indexing     : "values" or "position" 
                     "position": use numpy-like position index
                     "values": indexing on axis values 
        - tol           : tolerance when looking for numerical values, e.g. to use nearest neighbor search, default `None`
        - keepdims : keep singleton dimensions
        - broadcast_arrays: True, by default, consistently with numpy
        
            if False, indexing with list or array of indices will behave like
            Matlab TM does, which means that it will index on each individual dimensions.
            (internally, any list or array of indices will be converted to a boolean index
            of values before slicing)

            If True, numpy rules are followed. Consider the following case:

            a = DimArray(np.zeros((4,4,4)))
            a[[0,0],[0,0],[0,0]]
            
            if broadcast_arrays is False, the result will be a 3-D array of shape 2 x 2 x 2
            if broadcast_arrays is True, the result will be a 1-D array of size 2

        - mode: "raise", "clip", "wrap"
            analogous to numpy.ndarray.take's mode parameter, only valid (for now) if indexing is 'position'

    Returns
    -------
        - DimArray object or python built-in type, consistently with numpy slicing

    Examples
    --------

    >>> from dimarray import DimArray
    >>> v = DimArray([[1,2,3],[4,5,6]], axes=[["a","b"], [10.,20.,30.]], dims=['d0','d1'], dtype=float) 
    >>> v
    dimarray: 6 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (3): 10.0 to 30.0
    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.]])

    Indexing via axis values (default)

    >>> a = v[:,10]   # python slicing method
    >>> a
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0'
    0 / d0 (2): a to b
    array([ 1.,  4.])
    >>> b = v.take(10, axis=1)  # take, by axis position
    >>> c = v.take(10, axis='d1')  # take, by axis name
    >>> d = v.take({'d1':10})  # take, by dict {axis name : axis values}
    >>> (a==b).all() and (a==c).all() and (a==d).all()
    True

    Indexing via integer index (indexing="position" or `ix` property)

    >>> np.all(v.ix[:,0] == v[:,10])
    True
    >>> np.all(v.take(0, axis="d1", indexing="position") == v.take(10, axis="d1"))
    True

    Multi-dimensional indexing

    >>> v["a", 10]  # also work with string axis
    1.0
    >>> v.take(('a',10))  # multi-dimensional, tuple
    1.0
    >>> v.take({'d0':'a', 'd1':10})  # dict-like arguments
    1.0

    Take a list of indices

    >>> a = v[:,[10,20]] # also work with a list of index
    >>> a
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 1.,  2.],
           [ 4.,  5.]])
    >>> b = v.take([10,20], axis='d1')
    >>> np.all(a == b)
    True

    Take a slice:

    >>> c = v[:,10:20] # axis values: slice includes last element
    >>> c
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 1.,  2.],
           [ 4.,  5.]])
    >>> d = v.take(slice(10,20), axis='d1') # `take` accepts `slice` objects
    >>> np.all(c == d)
    True
    >>> v.ix[:,0:1] # integer position: does *not* include last element
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (1): 10.0 to 10.0
    array([[ 1.],
           [ 4.]])

    Keep dimensions 

    >>> a = v[["a"]]
    >>> b = v.take("a",keepdims=True)
    >>> np.all(a == b)
    True

    tolerance parameter to achieve "nearest neighbour" search

    >>> v.take(12, axis="d1", tol=5)
    dimarray: 2 non-null elements (0 null)
    dimensions: 'd0'
    0 / d0 (2): a to b
    array([ 1.,  4.])

    # Matlab like multi-indexing

    >>> v = DimArray(np.arange(2*3*4).reshape(2,3,4))
    >>> v.box[[0,1],:,[0,0,0]].shape
    (2, 3, 3)
    >>> v.box[[0,1],:,[0,0]].shape # here broadcast_arrays = False
    (2, 3, 2)
    >>> v[[0,1],:,[0,0]].shape # that is traditional numpy, with broadcasting on same shape
    (2, 3)
    >>> v.values[[0,1],:,[0,0]].shape # a proof of it
    (2, 3)

    >>> a = DimArray(np.arange(2*3).reshape(2,3))

    >>> a[a > 3] # FULL ARRAY: return a numpy array in n-d case (at least for now)
    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0,x1'
    0 / x0,x1 (2): (1, 1) to (1, 2)
    array([4, 5])

    >>> a[a.x0 > 0] # SINGLE AXIS: only first axis
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (1): 1 to 1
    1 / x1 (3): 0 to 2
    array([[3, 4, 5]])

    >>> a[:, a.x1 > 0] # only second axis 
    dimarray: 4 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (2): 1 to 2
    array([[1, 2],
           [4, 5]])

    >>> a.box[a.x0 > 0, a.x1 > 0]  # AXIS-BASED (need `box` to prevent broadcasting)
    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (1): 1 to 1
    1 / x1 (2): 1 to 2
    array([[4, 5]])

    Ommit `indices` parameter when putting a DimArray

    >>> a = DimArray([0,1,2,3,4], ['a','b','c','d','e'])
    >>> b = DimArray([5,6], ['c','d'])
    >>> a.put(b)
    dimarray: 5 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (5): a to e
    array([0, 1, 5, 6, 4])

    Ellipsis (only one supported)

    >>> a = DimArray(np.arange(2*3*4*5).reshape(2,3,4,5))
    >>> a[0,...,0].shape
    (3, 4)
    >>> a[...,0,0].shape
    (2, 3)
    """
    assert indexing in ("position", "values"), "invalid mode: "+repr(indexing)

    # SPECIAL CASE: full scale boolean array
    if obj.ndim > 1 and is_boolean_index(indices, obj.shape):
        indices = np.where(np.asarray(indices))
        newvalues = obj.values[indices]

        # return a scalar if size is 1
        if np.size(newvalues) <= 1:
            return newvalues

        # or return a DimArray with axes as tuple
        newaxisvalues = zip(*[obj.axes[i].values[ii] for i, ii in enumerate(indices)])
        newaxisname = ",".join(obj.dims)
        newaxis = Axis(newaxisvalues, newaxisname)
        newobj = obj._constructor(newvalues, [newaxis], **obj._metadata)
        return newobj

    indices = _fill_ellipsis(indices, obj.ndim)

    indices_numpy = obj.axes.loc(indices, axis=axis, position_index=(indexing == "position"), keepdims=keepdims, tol=tol)
    #try:
    #    indices_numpy = obj.axes.loc(indices, axis=axis, position_index=(indexing == "position"), keepdims=keepdims, tol=tol)
    #except IndexError, msg:
    #    raise IndexError(msg)

    # broadcast array-indices & integers, numpy's classical
    if broadcast_arrays:
        return _take_broadcast(obj, indices_numpy)

    # matlab-like, do not broadcast array-indices but simply sample values along each dimension independently
    else:
        return _take_box(obj, indices_numpy)

def _take_broadcast(a, indices):
    """ broadcast array-indices & integers, numpy's classical

    Examples
    --------
    >>> import dimarray as da
    >>> a = da.zeros(shape=(3,4,5,6))
    >>> a[:,[0, 1],:,2].shape
    (2, 3, 5)
    >>> a[:,[0, 1],2,:].shape
    (3, 2, 6)
    """
    # new values
    newval = a.values[indices]  

    # if the new values is a scalar, then just return it
    if np.isscalar(newval):
        return newval

    # new axes: broacast indices (should do the same as above, since integers are just broadcast)
    indices2 = broadcast_indices(indices)
    # assert np.all(newval == a.values[indices2])

    # make a multi-axis with tuples
    is_array2 = np.array([np.iterable(ix) for ix in indices2])
    nb_array2 = is_array2.sum()

    # If none or one array is present, easy
    if nb_array2 <= 1:
        newaxes = [a.axes[i][ix] for i, ix in enumerate(indices) if not np.isscalar(ix)] # indices or indices2, does not matter

    # else, finer check needed
    else:
        # same stats but on original indices
        is_array = np.array([np.iterable(ix) for ix in indices])
        array_ix_pos = np.where(is_array)[0]

        # Determine where the axis will be inserted
        # - need to consider the integers as well (broadcast as arrays)
        # - if two indexed dimensions are not contiguous, new axis placed at first position...
        # a = zeros((3,4,5,6))
            # a[:,[1,2],:,0].shape ==> (2, 3, 5)
            # a[:,[1,2],0,:].shape ==> (3, 2, 6)
        array_ix_pos2 = np.where(is_array2)[0]
        if np.any(np.diff(array_ix_pos2) > 1):  # that mean, if two indexed dimensions are not contiguous
            insert = 0
        else: 
            insert = array_ix_pos2[0]

        # Now determine axis value
        # ...if originally only one array was provided, use these values correspondingly
        if len(array_ix_pos) == 1:
            i = array_ix_pos[0]
            values = a.axes[i].values[indices[i]]
            name = a.axes[i].name

        # ...else use a list of tuples
        else:
            values = zip(*[a.axes[i].values[indices2[i]] for i in array_ix_pos])
            name = ",".join([a.axes[i].name for i in array_ix_pos])

        broadcastaxis = Axis(values, name)

        newaxes = Axes()
        for i, ax in enumerate(a.axes):

            # axis is already part of the broadcast axis: skip
            if is_array2[i]:
                continue

            else:
                newaxis = ax[indices2[i]]

                ## do not append axis if scalar
                #if np.isscalar(newaxis):
                #    continue

            newaxes.append(newaxis)

        # insert the right new axis at the appropriate position
        newaxes.insert(insert, broadcastaxis)

    return a._constructor(newval, newaxes, **a._metadata)

def _take_box(a, indices):
    """ matlab-like, do not broadcast array-indices but simply sample values along each dimension independently

    a: DimArray
    indices: numpy indices
    """
    # If there any array index?
    any_array = np.any([np.iterable(ix) for ix in indices])

    # convert each index to iterable if broadast_arrays is False
    if not any_array:
        return _take_broadcast(a, indices)

    # case where at least one array is present
    # e.g. np.zeros((2,2,2))[0,:,[0,0,0]] is of dimension 3 x 2, because the first dimension is broadcast to 3
    indices_ = ix_(indices, a.shape)

    # new numpy values
    newval = a.values[indices_] 

    # need to squeeze these dimensions where original index is an integer
    for i in reversed(range(len(indices))):
        if np.isscalar(indices[i]):
            newval = np.squeeze(newval, axis=i)

    # if scalar, just return it
    if np.isscalar(newval):
        return newval

    # New axes
    newaxes = [a.axes[i][ix] for i, ix in enumerate(indices) if not np.isscalar(ix)] # indices or indices2, does not matter

    return a._constructor(newval, newaxes, **a._metadata)

def _put(obj, val_, indices_numpy_, inplace=False, convert=False):
    """ put values in a DimArray using numpy integers, in the numpy way
    """
    if not inplace:
        obj = obj.copy()

    try:
        obj.values[indices_numpy_] = val_

    # convert to val.dtype if needed
    except ValueError, msg:
        if not convert: raise
        dtype = np.asarray(val_).dtype
        if obj.dtype is not dtype:
            obj.values = np.asarray(obj.values, dtype=dtype)
        obj.values[indices_numpy_] = val_

    if not inplace:
        return obj

def put(obj, val, indices=None, axis=0, indexing="values", tol=TOLERANCE, convert=False, inplace=False, broadcast_arrays=True):
    """ Put new values into DimArray (inplace)

    Parameters
    ----------
    obj: DimArray (do not provide if method bound to class instance)
    val: value to put in: scalar or array-like with appropriate shape or DimArray
    indices, optional: see `take` for indexing rules. indices may be omitted if 
        val is a DimArray (will then deduce `indices` from its axes)
    axis: for single index (see help on `take`)
    indexing : "position", "values"
    convert: convert array to val's type
    inplace: True
    broadcast_arrays: See documentation on `take`.

    Returns
    -------
    None: (inplace modification)

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray(np.zeros((2,2)), [('d0',['a','b']), ('d1',[10.,20.])])

    Index by values

    >>> b = a.put(1, indices={'d0': 'b'})
    >>> b
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  0.],
           [ 1.,  1.]])
    >>> a['b'] = 2   # slicing equivalent
    >>> a
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  0.],
           [ 2.,  2.]])

    Index by position

    >>> b = a.put(3, indices=1, axis='d1', indexing="position")
    >>> b
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  3.],
           [ 2.,  3.]])
    >>> a.ix[:,1] = 4  # slicing equivalent
    >>> a
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 2.,  4.]])


    Multi-dimension, multi-index

    >>> b = a.put(5, indices={'d0':'b', 'd1':[10.]})
    >>> b
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 5.,  4.]])
    >>> a["b",[10]] = 6
    >>> a
    dimarray: 4 non-null elements (0 null)
    dimensions: 'd0', 'd1'
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 6.,  4.]])

    Inplace

    >>> a.put(6, indices={'d0':'b', 'd1':[10.]}, inplace=True)

    Multi-Index tests (not straightforward to get matlab-like behaviour)

    >>> big = DimArray(np.zeros((2,3,4,5)))
    >>> indices = {'x0':0 ,'x1':[2,1],'x3':[1,4]}
    >>> sub = big.take(indices, broadcast_arrays=False)*0
    >>> sub.dims == ('x1','x2','x3')
    True
    >>> sub.shape == (2,4,2)
    True
    >>> big.put(sub+1, indices, inplace=True, broadcast_arrays=False)
    >>> sub2 = big.take(indices, broadcast_arrays=False)
    >>> np.all(sub+1 == sub2)
    True
    """
    assert indexing in ("position", "values"), "invalid mode: "+repr(indexing)

    if indices is None:

        # DimArray as subarray: infer indices from its axes
        if is_DimArray(val):
            indices = {ax.name:ax.values for ax in val.axes}
            broadcast_arrays = False  

        elif np.isscalar(val):
            raise ValueError("indices must be provided for non-DimArray or non-matching shape. See also `fill` method.")

        else:
            raise ValueError("indices must be provided for non-DimArray or non-matching shape")

    else:
            indices = _fill_ellipsis(indices, obj.ndim)


    # SPECIAL CASE: full scale boolean array
    if is_boolean_index(indices, obj.shape):
        return _put(obj, val, np.asarray(indices), inplace=inplace, convert=convert)

    indices_numpy = obj.axes.loc(indices, axis=axis, position_index=(indexing == "position"), tol=tol)

    # do nothing for full-array, boolean indexing
    #if len(indices_numpy) == 1 and isinstance

    # Convert to matlab-like indexing
    if not broadcast_arrays:

        indices_array = array_indices(indices_numpy, obj.shape)
        indices_numpy_ = np.ix_(*indices_array)
        shp = [len(ix) for ix in indices_array] # get an idea of the shape

        ## ...first check that val's shape is consistent with originally required indices
        # if DimArray, transpose to the right shape
        if is_DimArray(val):
            newdims = [d for d in obj.dims if d in val.dims] + [d for d in val.dims if d not in obj.dims]
            val = val.transpose(newdims)

        # only check for n-d array of size and dimensions > 1
        if np.size(val) > 1 and np.ndim(val) > 1 and np.any(np.array(shp) > 1):
            shp1 = [d for d in shp if d > 1]
            shp2 = [d for d in np.shape(val) if d > 1]
            if shp1 != shp2:
                raise ValueError('array is not broadcastable to correct shape (got values: {} but inferred from indices {})'.format(shp2, shp1))
    #
    #    # ...then reshape to new matlab-like form
        if np.isscalar(val):
            val_ = val
        elif np.size(val) == 1:
            val_ = np.squeeze(val)
        else:
            val = np.asarray(val)
            val_ = np.reshape(val, shp)

    else:
        val_ = val
        indices_numpy_ = indices_numpy

    return _put(obj, val_, indices_numpy_, inplace=inplace, convert=convert)

def fill(obj, val):
    """ anologous to numpy's fill (in-place operation)
    """
    obj.values.fill(val) 

#
# Variants 
#
def take_na(obj, indices, axis=0, indexing="values", tol=TOLERANCE, keepdims=False, fill_value=np.nan, repna=True, broadcast_arrays=True):
    """ like take but replace any missing value with NaNs

    additional parameters:
    na : replacement value, by default np.nan
    """
    #assert not broadcast_arrays, "check potential problem with array broadcasting"
    assert indexing in ("position", "values"), "invalid mode: "+repr(indexing)
    indices_numpy = obj.axes.loc(indices, axis=axis, position_index=(indexing == "position"), keepdims=keepdims, tol=tol, raise_error=not repna)

    indices_numpy, indices_mask = _filter_bad_indices(indices_numpy, obj.dims)
    result = take(obj, indices_numpy, indexing="position", broadcast_arrays=broadcast_arrays)
#    if np.isscalar(result):
#        return result if indices_mask is None else na
    put(result, fill_value, indices_mask, convert=True, inplace=True, tol=tol, indexing="position", broadcast_arrays=broadcast_arrays)

    # correct axis values
    if indices_mask is not None:

        #multi_indices = make_multiindex(indices, len(obj.axes)) # tuple of appropriate size
        indices = np.index_exp[indices] # make sure we have a tuple
        for i, dim in enumerate(indices_mask):
            ix = indices_mask[dim]
            result.axes[dim].values[ix] = np.asarray(indices[i])[ix]

    return result

def _filter_bad_indices(multi_index, dims):
    """ replace NaN indices by zeros

    Parameters
    ----------
        multi_index: tuple of `int` or `slice` objects

    Returns
    -------
        indices_nanfree: same as multi_index with NaNs replaced with 0
        indices_mask: multi-index of NaNs for the sliced array
        bad_ix: `dict` of `bool` 1-D arrays to indicate the locations of bad indices
    """
    multi_index = np.index_exp[multi_index]

    indices_nanfree = list(multi_index)
    indices_mask = {}

    for i, indices in enumerate(multi_index):
        dim = dims[i]

        if indices is None: 
            indices_mask[dim] = slice(None) # all NaNs

        if type(indices) is slice: 
            continue

        if type(indices) is int: 
            continue

        assert np.iterable(indices)

        bad = np.array([ix is None for ix in indices])
        valid_indices = np.zeros_like(bad, dtype=int)
        valid_indices[~bad] = [indices[k] for k, skip in enumerate(bad) if not skip]

        indices_mask[dim] = bad
        indices_nanfree[i] = valid_indices

    return tuple(indices_nanfree), indices_mask

###
### INTERP MODE?
###
##
##    # Pick-up values 
##    if mode == "interp":
##        result = _take_interp(obj, indices_numpy)
##
##
##
##
### Pick-up values 
##if mode == "interp":
##    result = _take_interp(obj, indices_numpy)
##
##

#    #
#    # add an extra "where" method
#    #
#    def where(self, condition, otherwise=None, axis=None):
#        """ 
#        parameters:
#        -----------
#
#            condition: bool array of same size as self, unless `axis=` is provided
#                 OR    `str` indicating a condition on axes
#            otherwise: array of same size as self or scalar, replacement value when condition is False
#            axis     : if provided, interpret the condition as applying along an axis
#
#        returns:
#        --------
#            
#            array with self values when condition is True, and `otherwise` if False
#            if only `condition` is provided, return axis values for which `condition` is True
#
#        Examples:
#        ---------
#            a.where(a > 0)
#        """
#        # convert scalar to the right shape
#        if np.size(otherwise) == 1:
#            otherwise += np.zeros_like(self.values)
#
#        # evaluate str condition
#        if type(condition) is str:
#            result = eval(condition, {ax.name:ax.values})
#
#        result = np.where(condition, [self.values, otherwise])

#
# Reindex axis
#

##    if position_index: repna = False # do not fill in with NaNs if position index
##
##    # Filter bad from good indices: this would happen only for repna=True
##    if repna:
##        indices_numpy, indices_mask = _filter_bad_indices(indices_numpy)
##
##    #
##    # SLICE or PICK ITEMS
##    #
##        
##
##    #
##    # NOW REPLACE MISSING DATA with NaNs
##    #
##    if repna and indices_mask is not None:
##        result = _fill_bad_indices(result, indices_mask, indices)

#    # fix type in interp mode
#    ix = np.asarray(ix) # make sure ix is an array
#    if not np.isscalar(result):
#        result.axes[axis].values = np.array(result.axes[axis].values, dtype=ix.dtype)

def reindex_axis(self, values, axis=0, method='exact', repna=True, fill_value=np.nan, tol=TOLERANCE, use_pandas=None):
    """ reindex an array along an axis

    Parameters
    ----------
    values : array-like or Axis: new axis values
    axis : axis number or name
    method : "exact" (default), "nearest", "interp" 
    repna : if False, raise error when an axis value is not present 
                 otherwise just replace with NaN. Defaulf is True
    fill_value: value to use instead of missing data
    tol: re-index with a particular tolerance (can be longer)
    use_pandas, optional: bool : if True (the default), convert to pandas for re-indexing 
      If any special option (method, tol) is set or if modulo axes are present 
      or, of course, if pandas is not installed,
      this option is set to False by default.

    Returns
    -------
    dimarray: DimArray instance

    Examples
    --------
    Basic reindexing: fill missing values with NaN

    >>> import dimarray as da
    >>> a = da.DimArray([1,2,3],('x0', [1,2,3]))
    >>> b = da.DimArray([3,4],('x0',[1,3]))
    >>> b.reindex_axis([1,2,3])
    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 1 to 3
    array([  3.,  nan,   4.])

    Or replace with anything else, like -9999

    >>> b.reindex_axis([1,2,3], fill_value=-9999)
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 1 to 3
    array([    3, -9999,     4])

    "nearest" mode

    >>> b.reindex_axis([0,1,2,3], method='nearest') # out-of-bound to NaN
    dimarray: 3 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (4): 0 to 3
    array([ nan,   3.,   3.,   4.])

    "interp" mode

    >>> b.reindex_axis([0,1,2,3], method='interp') # out-of-bound to NaN
    dimarray: 3 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (4): 0 to 3
    array([ nan,  3. ,  3.5,  4. ])
    """
    if isinstance(values, Axis):
        newaxis = values
        values = newaxis.values
        axis = newaxis.name

    axis_id = self.axes.get_idx(axis)
    ax = self.axes[axis_id] # Axis object
    axis_nm = ax.name

    # do nothing if axis is same or only None element
    if ax.values[0] is None or np.all(values==ax.values):
        return self

    # check whether pandas can be used for re-indexing
    if use_pandas is None:
        use_pandas = get_option('optim.use_pandas')

    # ...any special option activated?
    if method != 'exact' or tol is not None or \
            ax.tol is not None or ax.modulo is not None \
            or self.ndim > 4:  # pandas defined up to 4-D
                use_pandas = False

    # ...is pandas installed?
    try:
        import pandas
    except ImportError:
        use_pandas = False

    # re-index using pandas
    if use_pandas:
        pandasobj = self.to_pandas()
        newpandas = pandasobj.reindex_axis(values, axis=axis_id, fill_value=fill_value)
        newobj = self.from_pandas(newpandas) # use class method from_pandas
        newobj._metadata = self._metadata    # add metadata back
        newobj.axes[axis_id].name = axis_nm  # give back original name

    # indices along which to sample
    elif method == "exact":
        newobj = take_na(self, values, axis=axis, repna=repna, fill_value=fill_value)

    elif method in ("nearest", "interp"):
        from interpolation import interp
        newobj = interp(self, values, axis=axis, method=method, repna=repna)

    else:
        raise ValueError("invalid reindex_axis method: "+repr(method))

    #assert np.all((np.isnan(ax0.values) | (ax0.values == ax1.values))), "pb when reindexing"
    return newobj


def _reindex_axes(self, axes, **kwargs):
    """ reindex according to a list of axes
    """
    obj = self
    newdims = [ax2.name for ax2 in axes]
    for ax in self.axes:
        if ax.name in newdims:
            newaxis = axes[ax.name].values
            obj = obj.reindex_axis(newaxis, axis=ax.name, **kwargs)

    return obj

def reindex_like(self, other, method='exact', **kwargs):
    """ reindex_like : re-index like another dimarray / axes instance

    Notes
    -----
    only reindex axes which are present in other

    Examples
    --------
    >>> import dimarray as da
    >>> b = da.DimArray([3,4],('x0',[1,3]))
    >>> c = da.DimArray([[1,2,3], [1,2,3]],[('x1',["a","b"]),('x0',[1, 2, 3])])
    >>> b.reindex_like(c, method='interp')
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 1 to 3
    array([ 3. ,  3.5,  4. ])
    """
    if hasattr(other, 'axes'):
        axes = other.axes
    elif isinstance(other, Axes):
        axes = other
    else:
        raise TypeError('expected DimArray or Axes, got {}: {}'.format(type(other), other))
    return _reindex_axes(self, axes, method=method, **kwargs)


def sort_axis(a, axis=0, key=None):
    """ sort an axis 

    Parameters
    ----------
    a : DimArray (this argument is pre-assigned when using as bound method)
    axis, optional: axis by position (int) or name (str) (default: 0)
    key, optional: function that is called on each axis label and 
        whose return value is used for sorting instead of axis label.
        Any other object with __getitem__ attribute may also be used as key,
        such as a dictionary.
        If None (the default), axis label is used for sorting.

    Returns
    --------
    sorted: new DimArray with sorted axis

    Examples
    --------
    Basic
    >>> from dimarray import DimArray
    >>> a = DimArray([10,20,30], labels=[2, 0, 1])
    >>> a
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 2 to 1
    array([10, 20, 30])

    >>> a.sort_axis()
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([20, 30, 10])

    >>> a.sort_axis(key=lambda x: -x)
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 2 to 0
    array([10, 30, 20])

    Multi-dimensional
    >>> a = DimArray([[10,20,30],[40,50,60]], labels=[[0, 1], ['a','c','b']])
    >>> a.sort_axis(axis=1)
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): a to c
    array([[10, 30, 20],
           [40, 60, 50]])
    """
    index = a.axes[axis].values

    # convert key to a function
    if not hasattr(key, '__call__') and hasattr(key, '__getitem__'):
        key = key.__getitem__

    if key is None:
        ii = index.argsort()
    else:
        ii = argsort(index, key)

    return a.take(ii, axis=axis, indexing='position')

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

