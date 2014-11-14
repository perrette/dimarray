""" Base classes
"""
from __future__ import absolute_import
from collections import OrderedDict as odict
import numpy as np
import copy
from .indexing import locate_one, locate_many
from .prettyprinting import repr_axis, repr_dataset

class GetSetDelAttrMixin(object):
    """ Class to overload the __getattr__, __setattr__, __delattr__
    functions for easy access to attrs and axes. It assumes the attributes
    are stored in an `attrs` dictionary attribute, and check for presence of axes
    in an `axes` attribute and `dims`.
    """
    __metadata_exclude__ = []

    def __getattr__(self, name):
        if name.startswith('_') or name in self.__metadata_exclude__:
            pass
        elif 'dims' in self.__class__.__dict__ and name in self.dims:
            return self.axes[name].values # return axis values
        elif name in self.attrs.keys():
            return self.attrs[name]
        raise AttributeError("{} object has no attribute {}".format(self.__class__.__name__, name))

    def __setattr__(self, name, value):
        if name.startswith('_') \
                or name in self.__metadata_exclude__ \
                or hasattr(self.__class__, name):
            object.__setattr__(self, name, value) # do nothing special
        elif hasattr(self, 'axes') and name in self.dims:
            self.axes[name][:] = value # modify axis values
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
            applicable for numerical axes only. 
        issorted: bool, optional
            If True, assume the axis is sorted with increasing values (faster search)
        mode: {'raise', 'clip'}, optional
            Only applicable if `val` is array-like ignored otherwise and mode == 'raise'.
            If `mode == 'clip'`, any label not present in the axis is clipped to 
            the nearest end of the array. For a sorted array, an integer 
            position will be returned that maintained the array sorted. Note this 
            can result in unexpected return values for unsorted arrays.
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
        """
        values = self.values[:]
        tol=tol or self._tol

        if type(val) is slice:
            start = locate_one(values, val.start, tol=tol, issorted=issorted) if val.start is not None else None
            stop = locate_one(values, val.stop, tol=tol, issorted=issorted)+1 if val.stop is not None else None
            matches = slice(start, stop, val.step)

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
                    raise IndexError("Some values where not found in the axis: {}.".format(mismatch))

        return matches

    def _repr(self, metadata=False):
        return repr_axis(self, metadata=metadata)

class AbstractAxes(object):
    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError("Cannot set attribute to an Axes object")
        object.__setattr__(self, name, value)

class AbstractHasAxes(AbstractHasMetadata):
    """ class to handle things related to axes, such as overloading __getattr__
    """
    @property
    def dims(self):
        return tuple([ax.name for ax in self.axes])

    @dims.setter
    def dims(self, newdims):
        if not np.iterable(newdims): 
            raise TypeError("new dims must be iterable")
        if not isinstance(newdims, dict):
            if len(newdims) != len(self.dims):
                raise ValueError("Can only rename all dimensions at once, unless a dictionary is provided")
            newdims = dict(zip(self.dims, newdims))
        for old in newdims.keys():
            print "rename dim:",self.axes[old].name,newdims[old]
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
            self.axes[i][:] = lab

    def _get_indices(self, indices, axis=None, indexing=None, tol=None, keepdims=False):
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
        if indices is None: 
            indices = slice(None)

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
                    indices[self.dims[k]] = indices[k]
                    del indices[k] 
                else:
                    if k not in self.dims:
                        raise ValueError("Dimension {} not found. Existing dimensions: {}".format(k, self.dims))
                
            indices = tuple(indices[d] if d in indices else slice(None) for d in self.dims)

        elif not isinstance(indices, tuple):
            indices = (indices,)

        # expand "..." Ellipsis if any
        if np.any([ix is Ellipsis for ix in indices]):
            indices = _fill_ellipsis(indices, self.ndim)

        # load each dimension as necessary
        indexer = ()
        for i, dim in enumerate(self.dims):
            if i >= len(indices):
                ix = slice(None)
            else:
                ix = indices[i]

            # in case of label-based indexing, need to read the whole dimension
            # and look for the appropriate values
            if indexing != 'position' and not (type(ix) is slice and ix == slice(None)):
                # find the index corresponding to the required axis value
                lix = ix
                ix = self.axes[dim].loc(lix, tol=tol)

            # numpy rule: a singleton list does not collapse the axis
            if keepdims and np.isscalar(ix):
                ix = [ix]

            indexer += (ix,)

        return indexer

class AbstractDimArray(AbstractHasAxes):

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

    _indexing = None
    _broadcast = False

    # The indexing machinery in functional form, 
    # to be called by __getitem__ with default arguments
    def _getitem(self, indices=None, axis=None, indexing=None, tol=None, broadcast=None, keepdims=False):
        """ Retrieve values from a DimArray

        Parameters
        ----------
        indices : int or list or slice (single-dimensional indices)
                   or a tuple of those (multi-dimensional)
                   or `dict` of {{ axis name : axis values }}
        axis : None or int or str, optional
            if specified and indices is a slice, scalar or an array, assumes 
            indexing is along this axis.
        indexing : {'label', 'position', None}, optional
            Indexing mode. 
               - "position": use numpy-like position index (default)
               - "label": indexing on axis labels
            If None, call get_option('indexing.by'), which defaults to 'label'
        tol : None or float or tuple or dict, optional
            tolerance when looking for numerical values, e.g. to use nearest 
            neighbor search, default `None`.
        keepdims : bool, optional 
            keep singleton dimensions (default False)
        broadcast : bool, optional
            if True, use numpy-like `fancy` indexing and broacast any 
            indexing array to a common shape, useful for example to sample
            points along a path. Default to False.

        Returns
        -------
        indexed_array : DimArray instance or scalar

        See Also
        --------
        DimArray.put, DimArrayOnDisk.read, DimArray.take_axis

        Examples
        --------

        >>> from dimarray import DimArray
        >>> v = DimArray([[1,2,3],[4,5,6]], axes=[["a","b"], [10.,20.,30.]], dims=['d0','d1'], dtype=float) 
        >>> v
        dimarray: 6 non-null elements (0 null)
        0 / d0 (2): a to b
        1 / d1 (3): 10.0 to 30.0
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]])

        Indexing via axis values (default)

        >>> a = v[:,10]   # python slicing method
        >>> a
        dimarray: 2 non-null elements (0 null)
        0 / d0 (2): a to b
        array([ 1.,  4.])
        >>> b = v.take(10, axis=1)  # take, by axis position
        >>> c = v.take(10, axis='d1')  # take, by axis name
        >>> d = v.take({{'d1':10}})  # take, by dict {{axis name : axis values}}
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
        >>> v.take({{'d0':'a', 'd1':10}})  # dict-like arguments
        1.0

        Take a list of indices

        >>> a = v[:,[10,20]] # also work with a list of index
        >>> a
        dimarray: 4 non-null elements (0 null)
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
        0 / d0 (2): a to b
        1 / d1 (2): 10.0 to 20.0
        array([[ 1.,  2.],
               [ 4.,  5.]])
        >>> d = v.take(slice(10,20), axis='d1') # `take` accepts `slice` objects
        >>> np.all(c == d)
        True
        >>> v.ix[:,0:1] # integer position: does *not* include last element
        dimarray: 2 non-null elements (0 null)
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
        0 / x0,x1 (2): (1, 1) to (1, 2)
        array([4, 5])

        >>> a[a.x0 > 0] # SINGLE AXIS: only first axis
        dimarray: 3 non-null elements (0 null)
        0 / x0 (1): 1 to 1
        1 / x1 (3): 0 to 2
        array([[3, 4, 5]])

        >>> a[:, a.x1 > 0] # only second axis 
        dimarray: 4 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        1 / x1 (2): 1 to 2
        array([[1, 2],
               [4, 5]])

        >>> a[a.x0 > 0, a.x1 > 0]
        dimarray: 2 non-null elements (0 null)
        0 / x0 (1): 1 to 1
        1 / x1 (2): 1 to 2
        array([[4, 5]])

        Sample points along a path, a la numpy

        >>> a.take(zip((0,1),(0,2),(1,2)), broadcast=True)

        Ommit `indices` parameter when putting a DimArray

        >>> a = DimArray([0,1,2,3,4], ['a','b','c','d','e'])
        >>> b = DimArray([5,6], ['c','d'])
        >>> a.put(b, inplace=False)
        dimarray: 5 non-null elements (0 null)
        0 / x0 (5): a to e
        array([0, 1, 5, 6, 4])

        Ellipsis (only one supported)

        >>> a = DimArray(np.arange(2*3*4*5).reshape(2,3,4,5))
        >>> a[0,...,0].shape
        (3, 4)
        >>> a[...,0,0].shape
        (2, 3)
        """
        if indices is None:
            indices = slice(None)
        if broadcast is None: 
            if self._broadcast is None:
                broadcast = get_option('indexing.broadcast')
            else: 
                broadcast = self._broadcast

        # special-case: full-shape boolean indexing (will fail with netCDF4)
        if self._is_boolean_index_nd(indices):
            values = self.values[np.asarray(indices)] # boolean index, just get it, or raise appropriate error
            if np.isscalar(values):
                return values
            idx = np.where(indices)
            axes = self._getaxes_broadcast(idx)
            dima = self._constructor(values, axes)
            dima.attrs.update(self.attrs)
            return dima

        # indices are defined per axis
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

    def _setitem(self, indices, values, axis=None, indexing=None, tol=None, broadcast=None, cast=False, inplace=True):
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
            self.values[np.asarray(indices)] = values # boolean index, just set it, or raise appropriate error

        else:
            idx = self._get_indices(indices, tol=tol, indexing=indexing, axis=axis)

            if broadcast:
                self._setvalues_broadcast(idx, np.asarray(values), cast=cast)
            else:
                self._setvalues_ortho(idx, np.asarray(values), cast=cast)

        if not inplace:
            return self

    # also use with [:] syntax, for default arguments
    __getitem__ = _getitem 
    __setitem__ = _setitem

    # orthogonal or broadcast indexing?
    def _setvalues_broadcast(self, idx_tuple, values, cast=False):
        raise NotImplementedError()

    def _getvalues_broadcast(self, idx_tuple):
        raise NotImplementedError()

    def _getaxes_broadcast(self, idx_tuple):
        raise NotImplementedError()

    def _setvalues_ortho(self, idx_tuple, values, cast=False):
        raise NotImplementedError()

    def _getvalues_ortho(self, idx_tuple):
        raise NotImplementedError()

    def _is_boolean_index_nd(self, idx):
        " check whether a[a > 2] kind of operation is intended, with a.ndim > 1 "
        return (hasattr(idx, 'dtype') and hasattr(idx, 'ndim')) \
            and idx.dtype.kind == 'b' and idx.ndim > 1

    def copy(self):
        raise NotImplementedError()

    @property
    def ix(self):
        " toggle between position-based and label-based indexing "
        newindexing = 'label' if self._indexing=='position' else 'position'
        new = copy.copy(self) # shallow copy, not to verwrite _indexing
        new._indexing = newindexing
        return new

    # after xray: add sel, isel, loc, iloc methods
    def sel(self, **indices):
        return self.loc[indices]

    def isel(self, **indices):
        return self.iloc[indices]

    @property
    def loc(self):
        return self if self._indexing == 'label' else self.ix

    @property
    def iloc(self):
        # return self if self._indexing == 'position' else self.ix
        return self if self._indexing == 'position' else self.ix

class AbstractDataset(AbstractHasAxes):
    _repr = repr_dataset
