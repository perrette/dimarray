import numpy as np

from dimarray.tools import is_DimArray
from axes import Axis, Axes
from indexing import ix_, _fill_ellipsis

# take.__doc__ = take.__doc__.format(broadcast_arrays=_doc_broadcast_arrays)


    # return obj._constructor(newval, newaxes, **obj._metadata())

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

def put(self, val, indices=None, axis=0, indexing="label", tol=None, convert=False, inplace=False, broadcast_arrays=True):
    """ Put new values into DimArray (inplace)

    Parameters
    ----------
    val : array-like
        values to put in the array
    indices : misc, optional
        see `DimArray.take` for indexing rules. indices may be omitted if 
        val is a DimArray (will then deduce `indices` from its axes)
    axis : int or str or None, optional
        for single index (see help on `DimArray.take`)
    indexing : "position" or "label", optional
        default is "label" for indexing by axis label instead of integer position on the axis
    convert : bool, optional
        convert array to val's type
    inplace : bool, optional
        If True, modify array values in place, otherwise make and return a 
        copy.  Default is False
    {broadcast_arrays}

    Returns
    -------
    None: (inplace modification)

    See Also
    --------
    DimArray.take

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray(np.zeros((2,2)), [('d0',['a','b']), ('d1',[10.,20.])])

    Index by values

    >>> b = a.put(1, indices={{'d0': 'b'}})
    >>> b
    dimarray: 4 non-null elements (0 null)
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  0.],
           [ 1.,  1.]])
    >>> a['b'] = 2   # slicing equivalent
    >>> a
    dimarray: 4 non-null elements (0 null)
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  0.],
           [ 2.,  2.]])

    Index by position

    >>> b = a.put(3, indices=1, axis='d1', indexing="position")
    >>> b
    dimarray: 4 non-null elements (0 null)
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  3.],
           [ 2.,  3.]])
    >>> a.ix[:,1] = 4  # slicing equivalent
    >>> a
    dimarray: 4 non-null elements (0 null)
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 2.,  4.]])


    Multi-dimension, multi-index

    >>> b = a.put(5, indices={{'d0':'b', 'd1':[10.]}})
    >>> b
    dimarray: 4 non-null elements (0 null)
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 5.,  4.]])
    >>> a["b",[10]] = 6
    >>> a
    dimarray: 4 non-null elements (0 null)
    0 / d0 (2): a to b
    1 / d1 (2): 10.0 to 20.0
    array([[ 0.,  4.],
           [ 6.,  4.]])

    Inplace

    >>> a.put(6, indices={{'d0':'b', 'd1':[10.]}}, inplace=True)

    Multi-Index tests (not straightforward to get matlab-like behaviour)

    >>> big = DimArray(np.zeros((2,3,4,5)))
    >>> indices = {{'x0':0 ,'x1':[2,1],'x3':[1,4]}}
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
    assert indexing in ("position", "values", "label"), "invalid mode: "+repr(indexing)

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
            indices = _fill_ellipsis(indices, self.ndim)


    # SPECIAL CASE: full scale boolean array
    if is_boolean_index(indices, self.shape):
        return _put(self, val, np.asarray(indices), inplace=inplace, convert=convert)

    indices_numpy = self.axes.loc(indices, axis=axis, position_index=(indexing == "position"), tol=tol)

    # do nothing for full-array, boolean indexing
    #if len(indices_numpy) == 1 and isinstance

    # Convert to matlab-like indexing
    if not broadcast_arrays:

        indices_array = array_indices(indices_numpy, self.shape)
        indices_numpy_ = np.ix_(*indices_array)
        shp = [len(ix) for ix in indices_array] # get an idea of the shape

        ## ...first check that val's shape is consistent with originally required indices
        # if DimArray, transpose to the right shape
        if is_DimArray(val):
            newdims = [d for d in self.dims if d in val.dims] + [d for d in val.dims if d not in self.dims]
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

    return _put(self, val_, indices_numpy_, inplace=inplace, convert=convert)

# put.__doc__ = put.__doc__.format(broadcast_arrays=_doc_broadcast_arrays)
#take.__doc__ = take.__doc__.format(broadcast_arrays=_doc_broadcast_arrays)

