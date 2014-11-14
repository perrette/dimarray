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
        matches = isort[indices]

    return matches

def isnumber(val):
    try:
        val+1
        if val == 1: pass # only scalar allowed
        return True

    except:
        return type(val) != bool

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


# #
# # Locate values on an axis
# #
# def locate(values, *args, **kwargs):
#     return Axis(values).loc(*args, **kwargs)
#
# class LocatorAxis(object):
#     """ This class is the core of indexing in dimarray. 
#
#         loc = LocatorAxis(values, **opt)  
#
#     where `values` represent the axis values
#
#
#     A locator instance is generated from within the Axis object, via 
#     its properties loc (valued-based indexing) and iloc (integer-based)
#
#         axis.loc  ==> LocatorAxis(values)  
#
#     A locator is hashable is a similar way to a numpy array, but also 
#     callable to update parameters on-the-fly.
#
#     It returns an integer index or `list` of `int` or `slice` of `int` which 
#     is understood by numpy's arrays. In particular we have:
#
#         loc[ix] == np.index_exp[loc[ix]][0]
#
#     The "set" method can also be useful for chained calls. We have the general 
#     equivalence:
#
#         loc(idx, **kwargs) :: loc.set(**kwargs)[idx]
#
#     """
#     _check_params = False # false  for multi indexing
#     def __init__(self, values, raise_error=True, position_index = False, keepdims = False, **opt):
#         """
#         values        : string list or numpy array
#
#         raise_error = True # raise an error if value not found?
#         """
#         # compatibility wiht other methods:
#         if 'indexing' in opt:
#             indexing = opt.pop('indexing')
#             assert indexing in ('values', 'position')
#             position_index = indexing == 'position'
#
#         self.values = values
#         self.raise_error = raise_error
#         self.position_index = position_index
#         self.keepdims = keepdims 
#         self._list = None   # store axis value as a list
#
#         # check parameter values (default to False)
# #        if self._check_params:
#         for k in opt: 
#             if not hasattr(self, k):
#                 if k in ('tol', 'modulo'): # need to clean that up in LocatorAxes
#                     pass
#                 else:
#                     raise ValueError("unknown parameter {} for {}".format(k, self.__class__))
#
#         assert not hasattr(self, 'indexing')
#
#         #self.__dict__.update(opt) # update default options
#
#     def tolist(self):
#         """ return axis values as a list
#         """
#         if self._list is None:
#             self._list = self.values.tolist()
#         return self._list
#
#     #
#     # wrapper mode: __getitem__ and __call__
#     #
#     def __getitem__(self, ix):
#         """ 
#         """
#         #
#         # check special cases
#         #
#         assert ix is not None, "index is None!"
#
#         if self.position_index:
#             return ix
#
#         # boolean indexing ?
#         if is_DimArray(ix):
#             ix = ix.values
#
#         if type(ix) in (np.ndarray,) and ix.dtype is np.dtype(bool):
#             return ix
#
#         # make sure (1,) is understood as 1 just as numpy would
#         elif type(ix) is tuple:
#             if len(ix) == 1:
#                 ix = ix[0]
#
#         #
#         # look up corresponding numpy indices
#         #
#         # e.g. 45:56
#         if type(ix) is slice:
#             res = self.slice(ix)
#
#         elif self._islist(ix):
#             res = map(self.locate, ix)
#             #res = [self.locate(i) for i in ix]
#
#         else:
#             res = self.locate(ix)
#
#         return res
#
#     def _islist(self, ix):
#         """ check if value is a list index (in the sense it will collapse an axis)
#         """
#         return type(ix) in (list, np.ndarray)
#
#     def __call__(self, ix, **kwargs):
#         """ general wrapper method
#         
#         Parameters
#         ----------
#         ix : int, list, slice, tuple (on integer index or axis values)
#         **kwargs: see help on LocatorAxis
#
#         Returns
#         -------
#             `int`, list of `int` or slice of `int`
#         
#         """
#         #if method is None: method = self.method
#         if len(kwargs) > 0:
#             self = self.set(**kwargs)
#
#         if self.keepdims and not self._islist(ix) and not type(ix) is slice:
#             ix = [ix]
#
#         return self[ix]
#
#     def set(self, **kwargs):
#         """ convenience function for chained call: update methods and return itself 
#         """
#         #self.method = method
#         dict_ = self.__dict__.copy()
#         dict_.update(kwargs)
#         return self.__class__(**dict_)
#
#     #
#     # locate single values
#     #
#     def locate(self, val):
#         """ locate with try/except checks
#         """
#         if not self._check_type(val):
#             raise TypeError("{}: locate: wrong type {} --> {}".format(self.__class__, type(val), val))
#
#         try:
#             res = self._locate(val)
#
#         except IndexError, msg:
#             if self.raise_error:
#                 raise
#             else:
#                 res = None
#
#         return res
#
#     def _check_type(self, val): 
#         return True
#
#     def _locate(self, val):
#         """ locate without try/except check
#         """
#         raise NotImplementedError("to be subclassed")
#
#     #
#     # Access a slice
#     #
#     def slice(self, slice_, include_last=True):
#         """ Return a slice_ object
#
#         Parameters
#         ----------
#         slice_ : slice or tuple 
#         include_last : include last element 
#
#         Notes
#         -----
#         Note bound checking is automatically done via "locate" mode
#         This is in contrast with slicing in numpy arrays.
#         """
#         # Check type
#         if type(slice_) is not slice:
#             raise TypeError("should be slice !")
#
#         start, stop, step = slice_.start, slice_.stop, slice_.step
#
#         if start is not None:
#             start = self.locate(start)
#             if start is None: raise ValueError("{} not found in: \n {}:\n ==> invalid slice".format(start, self.values))
#
#         if stop is not None:
#             stop = self.locate(stop)
#             if stop is None: raise ValueError("{} not found in: \n {}:\n ==> invalid slice".format(stop, self.values))
#             
#             #at this stage stop is an integer index on the axis, 
#             # so make sure it is included in the slice if required
#             if include_last:
#                 stop += 1
#
#         # leave the step unchanged: it always means subsampling
#         return slice(start, stop, step)
#
# class ObjLocator(LocatorAxis):
#     """ locator axis for strings
#     """
#     def _locate(self, val):
#         """ find a string
#         """
#         try:
#             return self.tolist().index(val)
#         except ValueError, msg:
#             raise IndexError(msg)
#
#
# class NumLocator(LocatorAxis):
#     """ Locator for axis of integers or floats to be treated as numbers (with tolerance parameters)
#
#     Examples
#     --------
#     >>> values = np.arange(1950.,2000.)
#     >>> values  # doctest: +ELLIPSIS
#     array([ 1950., ... 1999.])
#     >>> loc = NumLocator(values)   
#     >>> loc(1951) 
#     1
#     >>> loc([1960, 1980, 1999])                # a list if also fine 
#     [10, 30, 49]
#     >>> loc(slice(1960,1970))                # or a tuple/slice (latest index included)
#     slice(10, 21, None)
#     >>> loc[1960:1970] == _                # identical, as any of the commands above
#     True
#     >>> loc([1960, -99, 1999], raise_error=False)  # handles missing values
#     [10, None, 49]
#
#     Test equivalence with np.index_exp
#     >>> ix = 1951
#     >>> loc[ix] == np.index_exp[loc[ix]][0]
#     True
#     >>> ix = [1960, 1980, 1999]
#     >>> loc[ix] == np.index_exp[loc[ix]][0]
#     True
#     >>> ix = slice(1960,1970)
#     >>> loc[ix] == np.index_exp[loc[ix]][0]
#     True
#     >>> ix = 1951
#     >>> loc[ix] == np.index_exp[loc[ix]][0]
#     True
#
#     # Modulo
#     >>> loc = NumLocator(np.array([0, 180, 360]), modulo=360)
#     >>> loc[180] == loc[-180]
#     True
#     """
#     def __init__(self, *args, **kwargs):
#
#         # extract parameters specific to NumLocator
#         opt = {'tol':None, 'modulo':None}
#         for k in kwargs.copy():
#             if k in opt:
#                 opt[k] = kwargs.pop(k)
#
#         super(NumLocator, self).__init__(*args, **kwargs)
#
#         self.__dict__.update(opt)
#         #print self.indexing
#
#     def _check_type(self, val):
#         return isnumber(val)
#
#     def _locate(self, val):
#         """ 
#         """
#         values = self.values
#
#         # modulo calculation, val = val +/- modulo*n, where n is an integer
#         # e.g. longitudes has modulo = 360
#         if self.modulo is not None:
#
#             if not isnumber(self.modulo):
#                 raise TypeError("modulo parameter need to be a number, got {} --> {}".format(type(self.modulo), self.modulo))
#                         
#             #mi, ma = values.min(), values.max() # min, max
#             mi, ma = self.min(), self.max() # min, max
#
#             if self.modulo and (val < mi or val > ma):
#                 val = _adjust_modulo(val, self.modulo, mi)
#
#         if self.tol is not None:
#
#             # locate value in axis
#             loc = np.argmin(np.abs(val-values))
#
#             if np.abs(values[loc]-val) > self.tol:
#                 raise IndexError("%f not found within tol %f (closest match %i:%f)" % (val, self.tol, loc, values[loc]))
#
#         else:
#             try:
#                 loc = self.tolist().index(val)
#             except ValueError, msg:
#                 raise IndexError("{}. Try setting axis `tol` parameter for nearest neighbor search.".format(msg))
#
#         return loc
#
#     def min(self):
#         return self.values.min()
#     def max(self):
#         return self.values.max()
#
# class RegularAxisLoc(NumLocator):
#     """ Locator for numerical axis with monotonically increasing, regularly spaced values
#     """
#     def min(self):
#         return self.values[0]
#
#     def max(self):
#         return self.values[-1]
#
#     @property
#     def step(self):
#         return self.values[1] - self.values[0]
#
#     
#
# # def is_boolean_index(indices, shape):
# #     """ check if something like a[a>2] is performed for ndim > 1
# #     """
# #     #indices = np.index_exp[indices]
# #     #if len(shape) > 1 and len(indices) == 1:
# #     if isinstance(indices, np.ndarray) or is_DimArray(indices):
# #         if indices.shape == shape:
# #             if indices.dtype == np.dtype(bool):
# #                 return True
# #
# #     return False # otherwise
#
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

    indices_numpy : multi-index
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

def ix_(indices_numpy, shape):
    # convert to matlab-like compatible indices
    """ convert numpy-like to matlab-like indices

    indices_numpy : indices to convert
    shape : shape of the array, to convert slices
    """
    dummy_ix = array_indices(indices_numpy, shape)
    return np.ix_(*dummy_ix)

#
# Functions to 
#
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
