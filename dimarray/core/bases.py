""" Base classes
"""
from __future__ import absolute_import
import warnings
import copy
from collections import OrderedDict as odict
import numpy as np
from dimarray.config import get_option
from dimarray.compat.pycompat import iteritems, zip
from dimarray.tools import is_numeric
from .indexing import locate_one, locate_many, locate_slice, expanded_indexer
from dimarray.prettyprinting import repr_axis, repr_dataset, repr_axes, str_axes, str_dataset, str_dimarray

class GetSetDelAttrMixin(object):
    """ Class to overload the __getattr__, __setattr__, __delattr__
    functions for easy access to attrs and axes. It assumes the attributes
    are stored in an `attrs` dictionary attribute, and check for presence of axes
    in an `axes` attribute and `dims`.
    """
    __metadata_exclude__ = [] # do not add to attrs
    __metadata_include__ = [] 

    def __getattr__(self, name):
        if hasattr(self.__class__, name):
            return object.__getattribute__(self, name)
        elif name not in self.__metadata_include__ \
                and (name.startswith('_') or name in self.__metadata_exclude__):
            pass # raise error
        elif hasattr(self, 'dims') and name in self.dims:
            return self.axes[name].values # return axis values
        elif name in self.attrs.keys():
            return self.attrs[name]
        raise AttributeError("{} object has no attribute {}".format(self.__class__.__name__, name))

    def __setattr__(self, name, value):
        if name not in self.__metadata_include__ and \
                (name.startswith('_') \
                 or name in self.__metadata_exclude__ \
                 or hasattr(self.__class__, name)):
            object.__setattr__(self, name, value) # do nothing special
        elif hasattr(self, 'axes') and name in self.dims:
            self.axes[name][()] = value # modify axis values
        else:
            self.attrs[name] = value # add as metadata

    def __delattr__(self, name):
        if not name.startswith('_') \
                and name not in self.__metadata_exclude__ \
                and not hasattr(self.__class__, name) \
                and name in self.attrs.keys():
            del self.attrs[name]
        else:
            return object.__delattr__(self, name)

# class GetSetDelAttrMixin(object):
#     pass
#
class AbstractHasMetadata(object):
       
    @property
    def attrs(self):
        return self._attrs # only for the in-memory array

    @attrs.setter
    def attrs(self, value):
        del self.attrs
        self.attrs.update(value)
         
    @attrs.deleter
    def attrs(self):
        for k in self.attrs.keys():
            del self.attrs[k]

    def _repr(self, metadata=False):
        return NotImplementedError()

    def __repr__(self):
        return self._repr()

    __str__ = str_dimarray

    def summary(self):
        print self.summary_repr()

    def summary_repr(self):
        return self._repr(metadata=True)

    def _metadata(self, meta=None):
        " for back compatibility "
        if meta is None:
            return self.attrs
        else:
            self.attrs.update(meta)


class AbstractAxis(AbstractHasMetadata):

    _tol = None

    def loc(self, val, tol=None, issorted=False, mode='raise'):
        """ Locate one or several values, using numpy functions

        Parameters
        ----------
        val: scalar or array-like or slice, optional
            The value(s) to look for in the axis.
        tol: None or float, optional
            If different from None, search nearest neighbour, with a certain 
            tolerance (np.argmin is used). None by default. This option is 
            applicable for numerical axes only. It does not apply for slices.
        issorted: bool, optional
            If True, assume the axis is sorted with increasing values (faster search)
        mode: {'raise', 'clip'}, optional
            Only applicable if `val` is array-like ignored otherwise and mode == 'raise'.
            If `mode=='clip'`, any label not present in the axis is clipped to 
            the nearest values (see np.searchsorted).
            If mode == 'raise' (the default), a check is performed on the result to ensure that
            all values were present, and raise an IndexError exception otherwise.

        Returns
        -------
        matches: integer position(s) of val in the axis

        Notes
        -----
        For single values and slices, non-zero elements in the array `a == val` are 
        searched for, and the first element is returned.
        For array-like indexer, the axis is sorted and np.searchsorted is applied.
        If tol is provided, a `np.argmin(np.abs(a-val))` is used.

        Examples
        --------
        >>> from dimarray import Axis
        >>> ax = Axis([1.5, 2.5, 3.5], 'myaxis')
        >>> ax.loc(2.5)
        1
        >>> ax.loc(slice(2, 3))
        slice(1, 2, None)
        >>> ax.loc(slice(2, 3.5))
        slice(1, 3, None)
        """
        values = self.values[()]
        tol=tol or self._tol

        if tol is not None and not self.is_numeric():
            tol = None # ignore tol parameter for non-numeric axes (an error will be raised if element is not found)

        if type(val) is slice:
            istart, istop = locate_slice(values, val.start, val.stop, val.step, issorted=issorted)
            matches = slice(istart, istop, val.step)

        elif np.isscalar(val):
            matches = locate_one(values, val, tol=tol, issorted=issorted)

        elif val is None:
            matches = values.tolist().index(val)

        elif hasattr(val, 'dtype') and val.dtype.kind == 'b':
            matches = val  # boolean indexing, do nothing

        elif tol is not None: # no scalar, but tolerance parameter provided
            matches = [locate_one(values, v, tol=tol, issorted=issorted) for v in val]

        else:
            matches = locate_many(values, val, issorted=issorted)

            if mode != 'clip':
                test = values[matches] != val
                mismatch = np.asarray(val)[test]
                if np.any(test):
                    raise IndexError("Some values where not found in the axis ({}): {}.".format(self.name, mismatch))

        return matches

    # def _repr(self, metadata=False):
    #     return repr_axis(self, metadata=metadata)
    _repr = repr_axis

    def __str__(self):
        """ simple string representation
        """
        #return "{}={}:{}".format(self.name, self.values[0], self.values[-1])
        return "{}({})={}:{}".format(self.name, self.size, *self._bounds())

    @property
    def dtype(self):
        return self.values.dtype

    def is_numeric(self):
        return is_numeric(self.values)

class AbstractAxes(object):
    _Axis = AbstractAxis
    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError("Cannot set attribute to an Axes object")
        object.__setattr__(self, name, value)

    __repr__ = repr_axes
    __str__ = str_axes

class Indexable(object):
    """Object to be indexed
    """
    def __init__(self, getitem, setitem, delitem, args=(), **kwargs):
        self.getitem = getitem
        self.setitem = setitem
        self.delitem = delitem
        self.args = args
        self.kwargs = kwargs
    def __getitem__(self, idx):
        return self.getitem(*self.args, indices=idx, **self.kwargs)
    def __setitem__(self, idx, item):
        return self.setitem(*self.args, indices=idx, values=item, **self.kwargs)
    def __delitem__(self, idx):
        return self.delitem(indices=idx, *self.args, **self.kwargs)

class AbstractHasAxes(AbstractHasMetadata):
    """ class to handle things related to axes, such as overloading __getattr__
    """
    _indexing = None
    _tol = None # define a tol attribute

    # indexing methods to be overloaded
    def _getitem(self, indices=None, **kwargs):
        raise NotImplementedError()
    def _setitem(self, indices=None, values=None, **kwargs):
        raise NotImplementedError()
    def _delitem(self, indices=None, **kwargs):
        raise NotImplementedError()

    @property
    def dims(self):
        return tuple([ax.name for ax in self.axes])

    @dims.setter
    def dims(self, newdims):
        self._set_dims(newdims)

    def _set_dims(self, newdims):
        if not np.iterable(newdims): 
            raise TypeError("new dims must be iterable")
        if not isinstance(newdims, dict):
            if len(newdims) != len(self.dims):
                raise ValueError("dimensions number mismatch")
            newdims = dict(zip(self.dims, newdims))
        for old in newdims.keys():
            self.axes[old].name = newdims[old]

    @property
    def axes(self):
        raise NotImplementedError('need to be overloaded !')

    @property
    def ndim(self):
        return len(self.axes)

    @property
    def labels(self):
        """ axis values 
        """
        return tuple([ax.values for ax in self.axes])

    @labels.setter
    def labels(self, newlabels):
        """ change all labels at once
        """
        if not np.iterable(newlabels): 
            raise TypeError("new labels must be iterable")
        if not len(newlabels) == self.ndim:
            raise ValueError("dimension mistmatch")
        for i, lab in enumerate(newlabels):
            self.axes[i][()] = lab

    def _get_indices(self, indices, axis=0, indexing=None, tol=None, keepdims=False):
        """ Return an n-D indexer  
        
        Parameters
        ----------
        **kwargs: same as DimArray.take or DimArrayOnDisk.read

        Returns
        -------
        indexer : tuple of numpy-compatible indices, of length equal to the number of 
            dimensions.
        """
        indexing = indexing or getattr(self,'_indexing',None) or get_option('indexing.by')
        dims = self.dims

        if indices is None:
            indices = ()

        if tol is None:
            tol = getattr(self, '_tol', None)

        #
        # Convert indices to tuple, from a variety of input formats
        #
        # special case: numpy like (idx, axis)
        if axis not in (0, None):
            indices = {axis:indices}

        # special case: Axes is provided as index
        elif isinstance(indices, AbstractAxes):
            indices = {ax.name:ax.values for ax in indices}

        # should always be a tuple
        if isinstance(indices, dict):
            # replace int dimensions with str dimensions
            for k in indices:
                if not isinstance(k, basestring):
                    indices[dims[k]] = indices[k]
                    del indices[k] 
                else:
                    if k not in dims:
                        raise ValueError("Dimension {} not found. Existing dimensions: {}".format(k, dims))
            indices = tuple(indices[d] if d in indices else slice(None) for d in dims)

        # expand to N-D tuple, and expands ellipsis
        indices = expanded_indexer(indices, self.ndim)

        # load each dimension as necessary
        indexer = ()
        for i, ix in enumerate(indices):
            dim = dims[i]

            if not np.isscalar(ix) and not isinstance(ix, slice):
                ix = np.asarray(ix)

            # boolean indices are fine
            if isinstance(ix, np.ndarray) and ix.dtype.kind == 'b':
                pass

            # in case of label-based indexing, need to read the whole dimension
            # and look for the appropriate values
            elif indexing != 'position' and not (type(ix) is slice and ix == slice(None)):
                # find the index corresponding to the required axis value
                lix = ix
                ix = self.axes[dim].loc(lix, tol=tol)

            # numpy rule: a singleton list does not collapse the axis
            if keepdims and np.isscalar(ix):
                ix = [ix]

            indexer += (ix,)

        return indexer

    def _getaxes_ortho(self, idx_tuple):
        " idx: tuple of position indices  of length = ndim (orthogonal indexing)"
        axes = []
        for i, ix in enumerate(idx_tuple):
            ax = self.axes[i][ix]
            if not np.isscalar(ax): # do not include scalar axes
                axes.append(ax)
        return axes

    #
    # returns axis position and name based on either of them
    #
    def _get_axis_info(self, axis):
        """ axis position and name

        Parameters
        ----------
        axis : `int` or `str` or None

        Returns
        -------
        idx : `int`, axis position
        name : `str` or None, axis name
        """
        if axis is None:
            return None, None

        if type(axis) in (str, unicode):
            idx = self.dims.index(axis)

        elif type(axis) is int:
            idx = axis

        else:
            raise TypeError("axis must be int or str, got:"+repr(axis))

        name = self.axes[idx].name
        return idx, name

    def _get_axes_info(self, axes):
        """ return axis (dimension) positions AND names from a sequence of axis (dimension) positions OR names

        Parameters
        ----------
        axes : sequence of str or int, representing axis (dimension) 
            names or positions, possibly mixed up.

        Returns
        -------
        pos : list of `int` indicating dimension's rank in the array
        names : list of dimension names
        """
        pos, names = zip(*[self._get_axis_info(x) for x in axes])
        return pos, names

    @property
    def ix(self):
        # " toggle between position-based and label-based indexing "
        # newindexing = 'label' if self._indexing=='position' else 'position'
        # new = copy.copy(self) # shallow copy, not to verwrite _indexing
        # new._indexing = newindexing
        indexing = 'position' if self._indexing != 'position' else 'label'
        return Indexable(self._getitem, self._setitem, self._delitem, indexing=indexing)

    # after xray: add sel, isel, loc, iloc methods
    def sel(self, **indices):
        return self.loc[indices]

    def isel(self, **indices):
        return self.iloc[indices]

    @property
    def loc(self):
        return Indexable(self._getitem, self._setitem, self._delitem, indexing='label')

    @property
    def iloc(self):
        # return self if self._indexing == 'position' else self.ix
        return Indexable(self._getitem, self._setitem, self._delitem, indexing='position')

    @property
    def nloc(self):
        # nearest neighbor loc
        return Indexable(self._getitem, self._setitem, self._delitem, indexing='label', tol=np.inf)


class OpMixin(object):
    """ overload basic operations
    """
    def _unary_op(self, func):
        raise NotImplementedError()
    def _binary_op(self, func, other):
        raise NotImplementedError()
    def _rbinary_op(self, func, other):
        return other._binary_op(func, self) # default only
    # def _cmp(self, func, other):
    #     return self._binary_op(func, other) # default only

    def __neg__(self): return self._unary_op(np.ndarray.__neg__)
    def __pos__(self): return self._unary_op(np.ndarray.__pos__)
    def __sqrt__(self, other): return self._unary_op(np.sqrt)
    def __invert__(self): return self._unary_op(np.invert)

    def __add__(self, other): return self._binary_op(np.add, other)
    def __sub__(self, other): return self._binary_op(np.subtract, other)
    def __mul__(self, other): return self._binary_op(np.multiply, other)

    def __div__(self, other): return self._binary_op(np.true_divide, other) # TRUE DIVIDE
    def __truediv__(self, other): return self._binary_op(np.true_divide, other)
    def __floordiv__(self, other): return self._binary_op(np.floor_divide, other)

    def __pow__(self, other): return self._binary_op(np.power, other)

    # reverse order operation
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return self._rbinary_op(np.subtract, other)
    def __rdiv__(self, other): return self._rbinary_op(np.true_divide, other)
    def __rpow__(self, other): return self._rbinary_op(np.power, other)



class AbstractDimArray(AbstractHasAxes):

    # @property
    # def tol(self):
    #     raise NotImplementedError()
    @property
    def values(self):
        raise NotImplementedError()
    @property
    def shape(self):
        return self.values.shape
    @property
    def ndim(self):
        return self.values.ndim
    @property
    def size(self):
        return self.values.size

    def is_numeric(self):
        return is_numeric(self.values)

    _broadcast = False

    # The indexing machinery in functional form, 
    # to be called by __getitem__ with default arguments
    def _getitem(self, indices=None, axis=0, indexing=None, tol=None, broadcast=None, keepdims=False,
                 broadcast_arrays=None,  # back-compatibility for broadcast
                 ):
        if indices is None:
            indices = ()

        if broadcast_arrays is not None:
            warnings.warn(FutureWarning("broadcast_arrays is deprecated, use broadcast instead"))
            broadcast = broadcast_arrays

        if broadcast is None: 
            if self._broadcast is None:
                broadcast = get_option('indexing.broadcast')
            else: 
                broadcast = self._broadcast

        # special-case: full-shape boolean indexing (will fail with netCDF4)
        if self._is_boolean_index_nd(indices):
            if hasattr(self, 'compress'):
                return self.compress(indices)
            else:
                raise TypeError("{} does not support boolean indexing".format(self.__class__.__name__))

        idx = self._get_indices(indices, axis=axis, indexing=indexing, tol=tol, keepdims=keepdims)

        # special case: broadcast arrays a la numpy
        if broadcast:
            axes = self._getaxes_broadcast(idx)
            values = self._getvalues_broadcast(idx)

        else:
            axes = self._getaxes_ortho(idx)
            values = self._getvalues_ortho(idx)

        if np.isscalar(values):
            return values

        dima = self._constructor(values, axes) # initialize DimArray
        dima.attrs.update(self.attrs) # add attribute

        return dima

    def _setitem(self, indices, values, axis=0, indexing=None, tol=None, broadcast=None, cast=False, inplace=True):
        """
        See Also
        --------
        DimArray.read, DimArrayOnDisk.write
        """
        if broadcast is None: 
            if self._broadcast is None:
                broadcast = get_option('indexing.broadcast')
            else: 
                broadcast = self._broadcast

        if not inplace:
            self = self.copy()

        # special-case: full-shape boolean indexing (will fail with netCDF4)
        if self._is_boolean_index_nd(indices):
            self._setvalues_bool(indices, values, cast=cast)

        else:
            idx = self._get_indices(indices, tol=tol, indexing=indexing, axis=axis)

            if broadcast:
                self._setvalues_broadcast(idx, values, cast=cast)
            else:
                self._setvalues_ortho(idx, values, cast=cast)

        if not inplace:
            return self

    __getitem__ = _getitem 
    __setitem__ = _setitem

    # def _getitem_1d(self, indices, axis=0, **kwargs):
    #     # by default, call _getitem (could be overloaded for optimization)
    #     return self._getitem({axis:indices}, **kwargs)

    # orthogonal or broadcast indexing?
    def _setvalues_broadcast(self, idx_tuple, values, cast=False):
        raise NotImplementedError()

    def _getvalues_broadcast(self, idx_tuple):
        raise NotImplementedError()

    def _getaxes_broadcast(self, idx_tuple):
        raise NotImplementedError()

    def _setvalues_ortho(self, idx_tuple, values, cast=False):
        raise NotImplementedError()

    def _setvalues_bool(self, mask, values, cast=False):
        raise NotImplementedError("boolean indexing is not implemented")

    def _getvalues_ortho(self, idx_tuple):
        raise NotImplementedError()

    def _is_boolean_index_nd(self, idx):
        " check whether a[a > 2] kind of operation is intended, with a.ndim > 1 "
        return (hasattr(idx, 'dtype') and hasattr(idx, 'ndim')) \
            and idx.dtype.kind == 'b' and idx.ndim > 1

    def copy(self):
        raise NotImplementedError()

class AbstractDataset(AbstractHasAxes):

    def _getitems(self, indices=None, axis=0, indexing=None, tol=None, broadcast=None, keepdims=False):

        # first find the index for the shared axes
        tuple_indices = self._get_indices(indices, axis=axis, tol=tol, keepdims=keepdims, indexing=indexing)

        # then index all arrays, one after the other
        newdata = self.__class__()

        # then apply take in 'position' mode
        newdata = self.__class__()

        axes_dict = {ax.name:ax[ix] for ix, ax in zip(tuple_indices, self.axes) if not np.isscalar(ix)}
        indices_dict = {ax.name:ix for ix, ax in zip(tuple_indices, self.axes)}

        # loop over variables
        for k in self.keys():
            v = self[k]
            # loop over axes to index on
            for axis in kw_indices.keys():
                if np.ndim(v) == 0 or axis not in v.dims: 
                    if raise_error: 
                        raise ValueError("{} does not have dimension {} ==> set raise_error=False to keep this variable unchanged".format(k, axis))
                    else:
                        continue
                # slice along one axis
                v = v.take({axis:kw_indices[axis]}, indexing='position')
            newdata[k] = v

        return newdata
    _repr = repr_dataset
    __str__ = str_dataset

# Add docstrings

