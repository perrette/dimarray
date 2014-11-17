""" Old indexing functions ==> imported from indexing.py. 
In the process of being rewritten.
"""
import functools
import copy 
import numpy as np
from dimarray.tools import is_DimArray
from dimarray.config import get_option
from dimarray.compat.pycompat import iteritems, range

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


def locate_one(values, val, issorted=False, tol=None, side='left'):
    " Locate one value in an axis"
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
        matches = np.searchsorted(values, val)
    else:
        isort = np.argsort(values)
        indices = np.searchsorted(values, val, sorter=isort, side=side)
        # if `val` is not found, and is larger than the other values, indices
        # is equal to values.size (therefore leading to an error)
        # mode="clip" solves this problem by returning the next value (the last)
        matches = isort.take(indices, mode='clip') 

    return matches

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
    """ update values's dtype so that values[:] = newval will not fail.
    """
    # check numpy-equivalent dtype
    dtype = np.asarray(newval).dtype

    # dtype comparison seems to be a good indicator of when type conversion works
    # e.g. dtype('O') > dtype(int) , dtype('O') > dtype(str) and dtype(float) > dtype(int) all return True
    # first convert Axis datatype to new values's type, if needed
    if values.dtype < dtype:
        values = np.asarray(values, dtype=dtype)

    # otherwise (no ordering relationship), just define an object type
    elif not (values.dtype  >= dtype):
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


def orthogonal_indexer(key, shape):
    """Given a key for orthogonal array indexing, returns an equivalent key
    suitable for indexing a numpy.ndarray with fancy indexing.
    """
    def expand_key(k, length):
        if isinstance(k, slice):
            return np.arange(k.start or 0, k.stop or length, k.step or 1)
        else:
            return k

    # replace Ellipsis objects with slices
    key = list(canonicalize_indexer(key, len(shape)))
    # replace 1d arrays and slices with broadcast compatible arrays
    # note: we treat integers separately (instead of turning them into 1d
    # arrays) because integers (and only integers) collapse axes when used with
    # __getitem__
    non_int_keys = [n for n, k in enumerate(key) if not isinstance(k, (int, np.integer))]

    def full_slices_unselected(n_list):
        def all_full_slices(key_index):
            return all(isinstance(key[n], slice) and key[n] == slice(None)
                       for n in key_index)
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

    array_indexers = np.ix_(*(expand_key(key[n], shape[n])
                              for n in array_keys))
    for i, n in enumerate(array_keys):
        key[n] = array_indexers[i]
    return tuple(key)
