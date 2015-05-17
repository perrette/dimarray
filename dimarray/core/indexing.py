""" Old indexing functions ==> imported from indexing.py. 
In the process of being rewritten.
"""
import warnings
import numpy as np
from dimarray.compat.pycompat import range

#__all__ = ["take", "put", "reindex_axis", "reindex_like"]
__all__ = []

#TOLERANCE=1e-8
TOLERANCE=None

_doc_broadcast_arrays = """
    broadcast_arrays : bool, optional
    
      False, by default.
    
      if False, indexing with list or array of indices will behave like
      Matlab TM does, which means that it will index on each individual dimensions.
      (internally, any list or array of indices will be converted to a boolean index
      of values before slicing)

      If True, numpy rules are followed. Consider the following case:

      a = DimArray(np.zeros((4,4,4)))
      a[[0,0],[0,0],[0,0]]
      
      if broadcast_arrays is False, the result will be a 3-D array of shape 2 x 2 x 2
      if broadcast_arrays is True, the result will be a 1-D array of size 2
      """.strip()


def _maybe_convert_datetime64(val):
    if isinstance(val, basestring): 
        try:
            val = np.datetime64(val)
        except Exception as error:
            warnings.warn(error.message)
    return val

def locate_one(values, val, issorted=False, tol=None, side='left'):
    " Locate one value in an axis"
    if values.dtype.kind == 'M': # 'datetime64'
        val = _maybe_convert_datetime64(val)

    if tol is not None:
        try:
            dist = np.abs(values - val)
            match = np.argmin(dist)
        except TypeError as error:
            raise TypeError("`tol` parameter only valid for numeric axes")
        if dist[match] > tol:
            raise IndexError("Did not find element `{}` in the axis with `tol={}`".format(repr(val), repr(tol)))

    elif issorted:
        match = np.searchsorted(values, val, side=side)

    else:
        matches = np.where(values == val)[0]
        if matches.size > 0:
            match = matches[0]
        else:
            raise IndexError("Element not found in axis: {}".format(repr(val)))

    return match

def locate_many(values, val, issorted=False, side='left'):
    """ Locate several values based on searchsorted
    """
    if issorted:
        matches = np.searchsorted(values, val, side=side)
    else:
        isort = np.argsort(values)
        indices = np.searchsorted(values, val, sorter=isort, side=side)
        # if `val` is not found, and is larger than the other values, indices
        # is equal to values.size (therefore leading to an error)
        # mode="clip" solves this problem by returning the next value (the last)
        matches = isort.take(indices, mode='clip') 

    return matches

def _locate_slice_strict(values, start, stop, step, issorted=False):
    " locate slice with strictly matching bounds"
    istart = locate_one(values, start, issorted=issorted) if start is not None else None
    istop = locate_one(values, stop, issorted=issorted) if stop is not None else None
    # include last element
    if stop is not None:
        istop += -1+2*(step is None or step>0)
    return istart, istop

def locate_slice(values, start, stop, step, issorted=False):
    """ Return integer indices for a slice

    For consistency with integer slices, and to make most use of slices 
    for sorted numerical axes (such as gridded data), 
    start and stop are searched for as a bounding box on the sorted numerical axis.
    start and stop are INCLUSIVE (stop belongs to the selected interval), for
    back-compatibility and compatibility with pandas.
    NOTE this is not the case for integer (axis position slices)
    """
    # special treatment for non-numeric (objects, boolean,...?) axes: just look for elements
    if not is_numeric(values):
        return _locate_slice_strict(values, start, stop, step, issorted=issorted)

    # bbox-like slices only make sense for monotonically varying axes
    if not issorted:
        monotonic = is_monotonic_equal(values)
        issorted = monotonic and values[-1] >= values[0]
    else:
        monotonic = True

    if not monotonic:
        return _locate_slice_strict(values, start, stop, step, issorted=issorted)

    # make sure that for numeric axes, the slice bounds are also numeric
    if start is not None and not is_numeric(np.asarray(start)):
        raise TypeError('numeric slice required for numeric axis')
    if stop is not None and not is_numeric(np.asarray(stop)):
        raise TypeError('numeric slice required for numeric axis')

    # At that point, consider slicing on monotonically varying numerical axes
    # ==> allow non-exact bounds, and interpret as a bounding box (inclusive of edges)
    if not issorted:
        inverted_axis = True
    else:
        inverted_axis = False

    if step is None or step > 0:
        right, left = 'right', 'left'
    else:
        right, left = 'left', 'right'

    if start is not None:
        if inverted_axis:
            istart = values.size - np.searchsorted(values[::-1], start, side=right)
        else:
            istart = np.searchsorted(values, start, side=left)

        if step is not None and step < 0:
            istart -= 1
    else:
        istart = None

    if stop is not None:
        if inverted_axis:
            istop = values.size - np.searchsorted(values[::-1], stop, side=left)
        else:
            istop = np.searchsorted(values, stop, side=right)

        if step is not None and step < 0:
            if istop == 0:
                istop = None   # 
            else:
                istop -= 1

    else:
        istop = None

    return istart, istop

#
# Check array ordering
#
def _is_ordered(values, cmp_):
    if values.size < 2:
        return True
    else:
        return np.all(cmp_(values[1:],values[:-1]))
def is_increasing(values):
    return _is_ordered(values, np.greater)
def is_increasing_equal(values):
    return _is_ordered(values, np.greater_equal)
def is_decreasing(values):
    return _is_ordered(values, np.less)
def is_decreasing_equal(values):
    return _is_ordered(values, np.less_equal)
def is_monotonic(values):
    return is_increasing(values) or is_decreasing(values)
def is_monotonic_equal(values):
    return is_increasing_equal(values) or is_decreasing_equal(values)

def is_numeric(values):
    return values.dtype.kind in ('f','i','u')

#
# Functions to 
#
def broadcast_indices(indices):
    """ Broadcast a tuple of indices onto a same shape.
    
    Notes
    -----
    If any array index is present, broadcast all arrays and integer indices on the same shape.
    This is the opposite of `ix_`.
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


def getaxes_broadcast(obj, indices):
    """ broadcast array-indices & integers, numpy's classical

    Examples
    --------
    >>> import dimarray as da
    >>> a = da.zeros(shape=(3,4,5,6))
    >>> a.take((slice(None),[0, 1],slice(None),2), broadcast=True).shape
    (2, 3, 5)
    >>> a.take((slice(None),[0, 1],2,slice(None)), broadcast=True).shape
    (3, 2, 6)
    """
    from dimarray import Axis, Axes

    # new axes: broacast indices (should do the same as above, since integers are just broadcast)
    indices2 = broadcast_indices(indices)
    # assert np.all(newval == obj.values[indices2])

    # make a multi-axis with tuples
    is_array2 = np.array([np.iterable(ix) for ix in indices2])
    nb_array2 = is_array2.sum()

    # If none or one array is present, easy
    if nb_array2 <= 1:
        newaxes = [obj.axes[i][ix] for i, ix in enumerate(indices) if not np.isscalar(ix)] # indices or indices2, does not matter

    # else, finer check needed
    else:
        # same stats but on original indices
        is_array = np.array([np.iterable(ix) for ix in indices])
        array_ix_pos = np.where(is_array)[0]

        # Determine where the axis will be inserted
        # - need to consider the integers as well (broadcast as arrays)
        # - if two indexed dimensions are not contiguous, new axis placed at first position...
        # obj = zeros((3,4,5,6))
            # obj[:,[1,2],:,0].shape ==> (2, 3, 5)
            # obj[:,[1,2],0,:].shape ==> (3, 2, 6)
        array_ix_pos2 = np.where(is_array2)[0]
        if np.any(np.diff(array_ix_pos2) > 1):  # that mean, if two indexed dimensions are not contiguous
            insert = 0
        else: 
            insert = array_ix_pos2[0]

        # Now determine axis value
        # ...if originally only one array was provided, use these values correspondingly
        if len(array_ix_pos) == 1:
            i = array_ix_pos[0]
            values = obj.axes[i].values[indices[i]]
            name = obj.axes[i].name

        # ...else use a list of tuples
        else:
            values = zip(*[obj.axes[i].values[indices2[i]] for i in array_ix_pos])
            name = ",".join([obj.axes[i].name for i in array_ix_pos])

        broadcastaxis = Axis(values, name)

        newaxes = Axes()
        for i, ax in enumerate(obj.axes):

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

    return newaxes


def _maybe_cast_type(values, newval):
    """ cast rules (chosen so that no information is lost), especially
    useful for axis values.
    # a[:] = b : a.dtype.kind
    # i <- f : f
    # U <- S : U
    # S <- U : U
    # O <- * : O
    # * <- O : O
    # b <- * : O
    # * <- b : O
    """
    # check numpy-equivalent dtype
    dtype = np.asarray(newval).dtype
    
    if values.dtype.kind == dtype.kind:
        pass # same kind
    elif values.dtype.kind == 'O':
        pass # or already object
    elif values.dtype.kind == 'f' and dtype.kind == 'i':
        pass # ok
    elif values.dtype.kind == 'i' and dtype.kind == 'f':
        values = np.asarray(values, dtype=float)
    elif values.dtype.kind == 'U' and dtype.kind == 'S':
        pass
    elif values.dtype.kind == 'S' and dtype.kind == 'U':
        values = np.asarray(values, dtype=unicode)
    else:
        values = np.asarray(values, dtype=object)

    return values

#####################################################################
#  The threee functions below have been taken from xray.core.indexing
#####################################################################

def expanded_indexer(key, ndim):
    """Given a key for indexing an ndarray, return an equivalent key which is a
    tuple with length equal to the number of dimensions.

    The expansion is done by replacing all `Ellipsis` items with the right
    number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    if not isinstance(key, tuple):
        # numpy treats non-tuple keys equivalent to tuples of length 1
        key = (key,)
    new_key = []
    # handling Ellipsis right is a little tricky, see:
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
    found_ellipsis = False
    for k in key:
        if k is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(k)
    if len(new_key) > ndim:
        raise IndexError('too many indices')
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)


def canonicalize_indexer(key, ndim):
    """Given an indexer for orthogonal array indexing, return an indexer that
    is a tuple composed entirely of slices, integer ndarrays and native python
    ints.
    """
    def canonicalize(indexer):
        if not isinstance(indexer, slice):
            indexer = np.asarray(indexer)
            if indexer.ndim == 0:
                indexer = int(np.asscalar(indexer))
            if isinstance(indexer, np.ndarray):
                if indexer.ndim != 1:
                    raise ValueError('orthogonal array indexing only supports '
                                     '1d arrays')
                if indexer.dtype.kind == 'b':
                    indexer, = np.nonzero(indexer)
                elif indexer.dtype.kind != 'i':
                    raise ValueError('invalid subkey %r for integer based '
                                     'array indexing; all subkeys must be '
                                     'slices, integers or sequences of '
                                     'integers or Booleans' % indexer)
        return indexer

    return tuple(canonicalize(k) for k in expanded_indexer(key, ndim))

def is_full_slice(value):
    return isinstance(value, slice) and value == slice(None)

def _expand_slice(slice_, size):
        return np.arange(*slice_.indices(size))

def orthogonal_indexer(key, shape):
    """Given a key for orthogonal array indexing, returns an equivalent key
    suitable for indexing a numpy.ndarray with fancy indexing.
    """
    # replace Ellipsis objects with slices
    key = list(canonicalize_indexer(key, len(shape)))
    # replace 1d arrays and slices with broadcast compatible arrays
    # note: we treat integers separately (instead of turning them into 1d
    # arrays) because integers (and only integers) collapse axes when used with
    # __getitem__
    non_int_keys = [n for n, k in enumerate(key) if not isinstance(k, (int, np.integer))]

    def full_slices_unselected(n_list):
        def all_full_slices(key_index):
            return all(is_full_slice(key[n]) for n in key_index)
        if not n_list:
            return n_list
        elif all_full_slices(range(n_list[0] + 1)):
            return full_slices_unselected(n_list[1:])
        elif all_full_slices(range(n_list[-1], len(key))):
            return full_slices_unselected(n_list[:-1])
        else:
            return n_list

    # However, testing suggests it is OK to keep contiguous sequences of full
    # slices at the start or the end of the key. Keeping slices around (when
    # possible) instead of converting slices to arrays significantly speeds up
    # indexing.
    # (Honestly, I don't understand when it's not OK to keep slices even in
    # between integer indices if as array is somewhere in the key, but such are
    # the admittedly mind-boggling ways of numpy's advanced indexing.)
    array_keys = full_slices_unselected(non_int_keys)

    def maybe_expand_slice(k, length):
        return _expand_slice(k, length) if isinstance(k, slice) else k

    array_indexers = np.ix_(*[maybe_expand_slice(key[n], shape[n])
                              for n in array_keys])
    for i, n in enumerate(array_keys):
        key[n] = array_indexers[i]
    return tuple(key)
