""" Old indexing functions ==> imported from indexing.py. 
In the process of being rewritten.
"""
import numpy as np
import functools
import copy 
from dimarray.tools import is_DimArray
from dimarray.config import get_option

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


def locate_one(values, val, issorted=False, tol=None):
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
        match = np.searchsorted(values, val)

    else:
        matches = np.where(values == val)[0]
        if matches.size > 0:
            match = matches[0]
        else:
            raise IndexError("Element not found in axis: {}".format(repr(val)))

    return match

def locate_many(values, val, issorted=False):
    """ Locate several values based on searchsorted
    """
    if issorted:
        matches = np.searchsorted(values, val)
    else:
        isort = np.argsort(values)
        indices = np.searchsorted(values, val, sorter=isort)
        # if `val` is not found, and is larger than the other values, indices
        # is equal to values.size (therefore leading to an error)
        # mode="clip" solves this problem by returning the next value (the last)
        matches = isort.take(indices, mode='clip') 

    return matches

# def isnumber(val):
#     try:
#         val+1
#         if val == 1: pass # only scalar allowed
#         return True
#
#     except:
#         return type(val) != bool

def _maybe_cast_type(values, newval):
    """ update values's dtype so that values[:] = newval will not fail.
    """
    # check numpy-equivalent dtype
    dtype = np.asarray(values).dtype

    # dtype comparison seems to be a good indicator of when type conversion works
    # e.g. dtype('O') > dtype(int) , dtype('O') > dtype(str) and dtype(float) > dtype(int) all return True
    # first convert Axis datatype to new values's type, if needed
    if values.dtype < dtype:
        values = np.asarray(values, dtype=dtype)

    # otherwise (no ordering relationship), just define an object type
    elif not (values.dtype  >= dtype):
        values = np.asarray(values, dtype=object)  

    return values


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

def ix_(indices, shape):
    """ Convert indices for orthogonal indexing

    Parameters
    ----------
    indices : (i0, i1, ...)
        can contains integer arrays, slice, integers
    shape : shape of the array, to convert slices

    Returns
    -------
    indices_ortho : (j0, j1, ...) 
        Indices so that for any compatible numpy array `a`, we have:
        `a[indices_ortho] == a[i0][:,i1][:,:,...]`

    Notes
    -----
    Singleton dimensions are maintained, need to squeeze the array accordingly
    """
    dummy_ix = []
    for i,ix in enumerate(indices):
        if not np.iterable(ix):
            if type(ix) is slice:
                ix = np.arange(shape[i])[ix]
            else:
                ix = np.array([ix]) # for np.ix_ to work
        # else make sure we have an array, and if bool, evaluate to arrays
        else:
            ix = np.asarray(ix)
            if ix.dtype.kind == 'b':
                ix = np.where(ix)[0]
        dummy_ix.append(ix)
    return np.ix_(*dummy_ix)

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
    >>> a[:,[0, 1],:,2].shape
    (2, 3, 5)
    >>> a[:,[0, 1],2,:].shape
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
