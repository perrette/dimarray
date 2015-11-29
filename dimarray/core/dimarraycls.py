#encoding:utf-8

""" array with physical dimensions (named and valued axes)
"""
from __future__ import absolute_import

import numpy as np
import copy
import warnings
from collections import OrderedDict as odict

from dimarray.tools import anynan, pandas_obj, format_doc
from dimarray.config import get_option
from dimarray import plotting
from dimarray.prettyprinting import repr_dimarray

# from .metadata import MetadataBase
from .bases import AbstractDimArray, GetSetDelAttrMixin, OpMixin
from .axes import Axis, Axes, MultiAxis
from .indexing import _maybe_cast_type, getaxes_broadcast, orthogonal_indexer
from .align import broadcast_arrays, align, stack

from . import transform as _transform  # numpy along-axis transformations, interpolation
from . import reshape as _reshape      # change array shape and dimensions
from . import operation as _operation  # operation between DimArrays
from . import missingvalues # operation between DimArrays
# from . import indexing as _indexing
from . import align as _align

__all__ = ["DimArray", "array"]

class DimArray(AbstractDimArray, OpMixin, GetSetDelAttrMixin):
    """ numpy's ndarray with labelled dimensions and axes

    Attributes
    ----------
    values : `ndarray`
    axes : `Axes` instance
        This is a custom list of axes.
        Each axis is an `Axis` instancce

    Dynamic attributes (properties):

    dims : tuple of axis names
    labels : tuple of axis values 
    
    shape, ndim, size, dtype : ndarray's attributes

    _metadata : `dict` of metadata (experimental)

    T : transposed DimArray
    ix : DimArray's view indexed by position index (numpy-like)

    Methods
    -------
    See online documentation

    Notes
    -----
    see interactive help for a full listing of methods with doc

    See Also
    --------
    Axes, Axis, MultiAxis, Dataset
    read_nc, stack, concatenate

    Examples
    --------

    >>> a = DimArray([[1.,2,3], [4,5,6]], axes=[['grl', 'ant'], [1950, 1960, 1970]], dims=['variable', 'time']) 
    >>> a
    dimarray: 6 non-null elements (0 null)
    0 / variable (2): 'grl' to 'ant'
    1 / time (3): 1950 to 1970
    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.]])

    Array data are stored in a `values` attribute:

    >>> a.values
    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.]])

    while its axes are stored in `axes`:

    >>> a.axes
    0 / variable (2): 'grl' to 'ant'
    1 / time (3): 1950 to 1970

    Each axis can be accessed by its rank or its name:

    >>> ax = a.axes[1]
    >>> ax.name , ax.values
    ('time', array([1950, 1960, 1970]))

    A few handy aliases are defined for the above, just like `shape`, `size` or `ndim`:

    >>> a.dims     # grab axis names (the dimensions)
    ('variable', 'time')

    >>> a.labels   # grab axis values
    (array(['grl', 'ant'], dtype=object), array([1950, 1960, 1970]))

    **Indexing works on axis values** instead of integer position:

    >>> a['grl', 1970]
    3.0

    but integer-index is always possible via `ix` toogle between
    `labels`- and `position`-based indexing:

    >>> a.ix[0, -1]
    3.0

    Standard numpy **transformations** are defined, and now accept axis name:

    >>> a.mean(axis='time')
    dimarray: 2 non-null elements (0 null)
    0 / variable (2): 'grl' to 'ant'
    array([ 2.,  5.])

    During an operation, arrays are automatically re-indexed to span the 
    same axis domain, with nan filling if needed

    >>> a = DimArray([0, 1], axes = [[0, 1]])
    >>> b = DimArray([0,1,2], axes = [[0, 1, 2]])
    >>> a+b
    dimarray: 2 non-null elements (1 null)
    0 / x0 (3): 0 to 2
    array([  0.,   2.,  nan])

    Dimensions are factored (broadcast) when performing an operation

    >>> a = DimArray([0, 1], dims=('x0',))
    >>> b = DimArray([0, 1, 2], dims=('x1',))
    >>> a+b
    dimarray: 6 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[0, 1, 2],
           [1, 2, 3]])
    """
    _order = None  # set a general ordering relationship for dimensions

    #
    # NOW MAIN BODY OF THE CLASS
    #

    def __init__(self, values=None, axes=None, dims=None, labels=None, copy=False, dtype=None, _indexing=None, _indexing_broadcast=None, **kwargs):
        """ Initialize a DimArray instance

        Parameters
        ----------

        values : numpy-like array, or DimArray instance, or dict 
                  If `values` is not provided, will initialize an empty array 
                  with dimensions inferred from axes (in that case `axes=` 
                  must be provided).

        axes : list or tuple, optional
        
            axis values as ndarrays, whose order 
            matches axis names (the dimensions) provided via `dims=` 
            parameter. Each axis can also be provided as a tuple 
            (str, array-like) which contains both axis name and axis 
            values, in which case `dims=` becomes superfluous.
            `axes=` can also be provided with a list of Axis objects
            If `axes=` is omitted, a standard axis `np.arange(shape[i])`
            is created for each axis `i`.

        dims : list or tuple, optional
            dimensions (or axis names)
            This parameter can be omitted if dimensions are already 
            provided by other means, such as passing a list of tuple 
            to `axes=`. If axes are passed as keyword arguments (via 
            **kwargs), `dims=` is used to determine the order of 
            dimensions. If `dims` is not provided by any of the means 
            mentioned above, default dimension names are 
            given `x0`, `x1`, ...`xn`, where n is the number of 
            dimensions.

        dtype : numpy data type, optional
            passed to np.array() 

        copy : bool, optional
            passed to np.array()

        **kwargs : keyword arguments
            metadata 
        
        Notes 
        ----- 
        metadata passed this way cannot have name already taken by other 
            parameters such as "values", "axes", "dims", "dtype" or "copy".

        Examples
        --------

        Basic:

        >>> DimArray([[1,2,3],[4,5,6]]) # automatic labelling
        dimarray: 6 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        1 / x1 (3): 0 to 2
        array([[1, 2, 3],
               [4, 5, 6]])

        >>> DimArray([[1,2,3],[4,5,6]], dims=['items','time'])  # axis names only
        dimarray: 6 non-null elements (0 null)
        0 / items (2): 0 to 1
        1 / time (3): 0 to 2
        array([[1, 2, 3],
               [4, 5, 6]])

        >>> DimArray([[1,2,3],[4,5,6]], axes=[list("ab"), np.arange(1950,1953)]) # axis values only
        dimarray: 6 non-null elements (0 null)
        0 / x0 (2): 'a' to 'b'
        1 / x1 (3): 1950 to 1952
        array([[1, 2, 3],
               [4, 5, 6]])

        More general case:

        >>> a = DimArray([[1,2,3],[4,5,6]], axes=[list("ab"), np.arange(1950,1953)], dims=['items','time']) 
        >>> b = DimArray([[1,2,3],[4,5,6]], axes=[('items',list("ab")), ('time',np.arange(1950,1953))])
        >>> c = DimArray([[1,2,3],[4,5,6]], {'items':list("ab"), 'time':np.arange(1950,1953)}) # here dims can be omitted because shape = (2, 3)
        >>> np.all(a == b) and np.all(a == c)
        True
        >>> a
        dimarray: 6 non-null elements (0 null)
        0 / items (2): 'a' to 'b'
        1 / time (3): 1950 to 1952
        array([[1, 2, 3],
               [4, 5, 6]])

        Empty data

        >>> a = DimArray(axes=[('items',list("ab")), ('time',np.arange(1950,1953))])

        Metadata

        >>> a = DimArray([[1,2,3],[4,5,6]], name='test', units='none') 
        
        """
        # check if attached to values (e.g. DimArray object)
        if hasattr(values, "axes") and axes is None:
            axes = values.axes

        self._attrs = odict(getattr(values, "attrs", {}))

        # default options
        if _indexing is None: _indexing = get_option('indexing.by')
        if _indexing_broadcast is None: _indexing_broadcast = get_option('indexing.broadcast')

        #
        # array values
        #
        # if masked array, replace mask by NaN
        if isinstance(values, np.ma.MaskedArray):
            try:
                values = values.filled(np.nan) # fill mask with nans

            # first convert to float
            except:
                values = np.ma.asarray(values, dtype=float).filled(np.nan) # fill mask with nans

        ## if not numpy ndarray or DimArray, call nested constructor
        ## e.g. nested dictionary, or lists and so on
        elif _contains_dictlike(values):
            if isinstance(axes, Axes) or np.iterable(axes) and len(axes)>0 and isinstance(axes[0], Axis):
                raise TypeError("if nested data please use dims and labels instead of axes")
            elif axes is not None:
                labels = axes

            dim_array = self.from_nested(values, dims=dims, labels=labels)
            values = dim_array.values
            axes = dim_array.axes

        elif values is not None:
            values = np.array(values, copy=copy, dtype=dtype)

        #
        # Initialize the axes
        # 
        if not isinstance(axes, Axes):
            axes = Axes._init(axes, dims=dims, labels=labels, shape=values.shape if values is not None else None)
        assert isinstance(axes, Axes)

        # if values not provided, create empty data, filled with NaNs if dtype is float
        if values is None:
            values = np.empty([ax.size for ax in axes], dtype=dtype)
            if dtype in (float, None, np.dtype(float)):
                values.fill(np.nan)
            else:
                warnings.warn("no nan representation for {}, array left empty".format(repr(dtype)))

        #
        # store all fields
        #
        self._attrs.update(kwargs)
        self._values = values
        self._axes = axes

        ## options
        self._indexing = _indexing
        self._indexing_broadcast = _indexing_broadcast

        # Check consistency between axes and values
        inferred = tuple([ax.size for ax in self.axes])
        if inferred != self.values.shape:
            msg = """\
shape inferred from axes: {}
shape inferred from data: {}
mismatch between values and axes""".format(inferred, self.values.shape)
            raise Exception(msg)

        # If a general ordering relationship of the class is assumed,
        # always sort the class
        if self._order is not None and self.dims != tuple(dim for dim in self._order if dim in self.dims):
            present = filter(lambda x: x in self.dims, self._order)  # prescribed
            missing = filter(lambda x: x not in self._order, self.dims)  # not
            order = missing + present # prepend dimensions not found in ordering relationship
            obj = self.transpose(order)
            self._values = obj.values
            self.axes = obj.axes

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, newvalues):
        self._values = _maybe_cast_type(self._values, newvalues)
        self._values[:] = newvalues

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, newaxes):
        if not isinstance(newaxes, Axes):
            newaxes = Axes._init(newaxes, shape=self.shape)
        else:
            assert [ax.size for ax in newaxes] == list(self.shape), "shape mismatch"
        self._axes = newaxes

    @property
    def dims(self):
        return tuple([ax.name for ax in self.axes])

    @dims.setter
    def dims(self, newdims):
        self._set_dims(newdims)

    @classmethod
    def from_nested(cls, nested_data, labels=None, dims=None, align=True):
        """ recursive definition of DimArray with nested lists or dict

        Parameters
        ----------
        nested_data : dict-like or list
            contains other nested data or nd.ndarray or DimArray (for recursive definition)
            or arrays or dimarrays
        labels : array-like, optional
            axis values
        dims : sequence of dimensions (axis names), optional
        align : bool, optional
            automatically align axes when stacking sub-arrays?
            default to True (see stack for help)

        Returns
        -------
        dim_array : DimArray instance

        Examples
        --------
        >>> nested_data = [{1:11,2:22,3:33} , {1:111,2:222,3:333}]
        >>> DimArray.from_nested(nested_data, dims=['dim1','dim2'])
        dimarray: 6 non-null elements (0 null)
        0 / dim1 (2): 0 to 1
        1 / dim2 (3): 1 to 3
        array([[ 11,  22,  33],
               [111, 222, 333]])

        Now also included in main DimArray

        >>> DimArray(nested_data, dims=['dim1','dim2'], labels=[['a','b']]) # enough to include the first level label in this case
        dimarray: 6 non-null elements (0 null)
        0 / dim1 (2): 'a' to 'b'
        1 / dim2 (3): 1 to 3
        array([[ 11,  22,  33],
               [111, 222, 333]])

        """
        if dims is None:
            dim0 = None
            subdims = None
        else:
            dim0 = dims[0]
            if len(dims) > 1:
                subdims = dims[1:]
            else:
                subdims = None

        if labels is None:
            label0 = None
            sublabels = None
        else:
            label0 = labels[0]
            if len(labels) > 1:
                sublabels = labels[1:]
            else:
                sublabels = None

        # if Dataset, create a dictionary from it
        from dimarray.dataset import Dataset
        if isinstance(nested_data, Dataset):
            nested_data = nested_data.to_dict()

        # if numpy ndarray, create a new DimArray
        # if DimArray, just update labels and return it
        elif isinstance(nested_data, DimArray) \
                or isinstance(nested_data, np.ndarray):
            return cls(nested_data, dims=dims, labels=labels)

        elif np.isscalar(nested_data):
            return cls(nested_data)

        # convert dict to sequence if needed
        if _is_dictlike(nested_data):
            if label0 is None:
                label0 = nested_data.keys()
            if callable(nested_data.values):
                nested_data = nested_data.values()
            else:
                nested_data = nested_data.values

        # Iterate over subarrays
        items = []
        for item in nested_data:
            if not isinstance(item, DimArray):
                item = cls.from_nested(item, labels=sublabels, dims=subdims)
            items.append(item)

        # stack sub-arrays
        dim_array = stack(items, keys=label0, axis=dim0, align=align)
        
        return dim_array

    #
    # Internal constructor, useful for subclassing
    #
    @classmethod
    def _constructor(cls, values, axes, **metadata):
        """ Internal API for the constructor: check whether a pre-defined class exists

        values        : array-like
        axes        : Axes instance 

        This static method is used whenever a new DimArray needs to be instantiated
        for example after a transformation.

        This makes the sub-classing process easier since only this method needs to be 
        overloaded to make the sub-class a "closed" class.

        """
        #TODO: use the __new__ operator to bypass all checkings in __init__
        # just check consistency between axes and values shape

        return cls(values, axes, **metadata)

    def copy(self, shallow=False):
        """ copy of the object and update arguments

        Parameters
        ----------
        shallow: if True, does not copy values and axes, only useful to overwrite
            attributes without affecting the initial array.

        Returns
        -------
        DimArray
        """
        import copy
        if shallow:
            new = copy.copy(self) # shallow copy
        else:
            new = copy.deepcopy(self) # shallow copy
        return new

    #
    # misc
    #
    @property
    def dtype(self): 
        return self.values.dtype

    @property
    def __array__(self): 
        """ so that np.array() works as expected (returns values)
        """
        return self.values.__array__

    def __array_wrap__(self, result): 
        """ returns a DimArray when doing e.g. self.values + self

        >>> a = DimArray([3,4], axes=[('xx',['a','b'])])
        >>> a.values + a
        dimarray: 2 non-null elements (0 null)
        0 / xx (2): 'a' to 'b'
        array([6, 8])
        """
        return self._constructor(result, self.axes) # copy=True by default, ok?

    #
    # iteration
    #

    def __iter__(self): 
        """ iterates on values along the first axis, consistently with a ndarray
        """
        for k, val in self.iter():
            yield val

    def iter(self, axis=0, keepdims=False):
        """ Iterate over axis value and cross-section, along any axis (by default the first)

        for time, time_slice in myarray.iter('time'):
            do stuff
        """
        # iterate over axis values
        for i, k in enumerate(self.axes[axis].values):
            val = self.take(i, axis=axis, indexing='position', keepdims=False) # cross-section
            yield k, val


    #
    # INDEXING
    #

    #
    # New general-purpose indexing method
    #
    @property
    def take(self):
        """ Retrieve values from a DimArray

        Parameters
        ----------
        indices : int or list or slice (single-dimensional indices)
                   or a tuple of those (multi-dimensional)
                   or `dict` of { axis name : axis values }
        axis : None or int or str, optional
            if specified and indices is a slice, scalar or an array, assumes 
            indexing is along this axis.
        indexing : {'label', 'position'}, optional
            Indexing mode. 
            - "label": indexing on axis labels (default)
            - "position": use numpy-like position index
            Default value can be changed in dimarray.rcParams['indexing.by']
        tol : None or float or tuple or dict, optional
            tolerance when looking for numerical values, e.g. to use nearest 
            neighbor search, default `None`.
        keepdims : bool, optional 
            keep singleton dimensions (default False)
        broadcast : bool, optional
            if True, use numpy-like `fancy` indexing and broadcast any 
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
        0 / d0 (2): 'a' to 'b'
        1 / d1 (3): 10.0 to 30.0
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]])

        Indexing via axis values (default)

        >>> a = v[:,10]   # python slicing method
        >>> a
        dimarray: 2 non-null elements (0 null)
        0 / d0 (2): 'a' to 'b'
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
        0 / d0 (2): 'a' to 'b'
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
        0 / d0 (2): 'a' to 'b'
        1 / d1 (2): 10.0 to 20.0
        array([[ 1.,  2.],
               [ 4.,  5.]])
        >>> d = v.take(slice(10,20), axis='d1') # `take` accepts `slice` objects
        >>> np.all(c == d)
        True
        >>> v.ix[:,0:1] # integer position: does *not* include last element
        dimarray: 2 non-null elements (0 null)
        0 / d0 (2): 'a' to 'b'
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
        0 / d0 (2): 'a' to 'b'
        array([ 1.,  4.])

        # Matlab like multi-indexing

        >>> v = DimArray(np.arange(2*3*4).reshape(2,3,4))
        >>> v[[0,1],:,[0,0,0]].shape
        (2, 3, 3)
        >>> v[[0,1],:,[0,0]].shape # here broadcast = False
        (2, 3, 2)
        >>> v.take(([0,1],slice(None),[0,0]), broadcast=True).shape # that is traditional numpy, with broadcasting on same shape
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

        Sample points along a path, a la numpy, with broadcast=True

        >>> a.take(([0,0,1],[1,2,2]), broadcast=True)
        dimarray: 3 non-null elements (0 null)
        0 / x0,x1 (3): (0, 1) to (1, 2)
        array([1, 2, 5])

        Ellipsis (only one supported)

        >>> a = DimArray(np.arange(2*3*4*5).reshape(2,3,4,5))
        >>> a[0,...,0].shape
        (3, 4)
        >>> a[...,0,0].shape
        (2, 3)
        """
        return self._getitem

    @property
    def put(self):
        """Modify values of a DimArray

        Parameters
        ----------
        indices : int or list or slice (single-dimensional indices)
                   or a tuple of those (multi-dimensional)
                   or `dict` of { axis name : axis values }
        axis : None or int or str, optional
            if specified and indices is a slice, scalar or an array, assumes 
            indexing is along this axis.
        indexing : {'label', 'position'}, optional
            Indexing mode. 
            - "label": indexing on axis labels (default)
            - "position": use numpy-like position index
            Default value can be changed in dimarray.rcParams['indexing.by']
        tol : None or float or tuple or dict, optional
            tolerance when looking for numerical values, e.g. to use nearest 
            neighbor search, default `None`.
        broadcast : bool, optional
            if True, use numpy-like `fancy` indexing and broadcast any 
            indexing array to a common shape, useful for example to sample
            points along a path. Default to False.

        Returns
        -------
        None (inplace=True) or DimArray instance or scalar (inplace=False)

        See Also
        --------
        DimArray.take, DimArrayOnDisk.write
        """
        return self._setitem

    def compress(self, boolarray):
        """ Boolean indexing
        """
        try:
            boolarray = np.asarray(boolarray)
            assert boolarray.dtype.kind == 'b'
        except:
            raise TypeError("Invalid dtype for boolean indexing: "+boolarray.dtype.kind)
        if boolarray.ndim != self.ndim:
            raise ValueError("Indexing array dimensions must match indexed array")
        values = self.values[boolarray] # boolean index, just get it, or raise appropriate error
        if np.isscalar(values):
            return values
        idx = np.where(boolarray)
        axes = self._getaxes_broadcast(idx)
        dima = self._constructor(values, axes)
        dima.attrs.update(self.attrs)
        return dima

    def compress_axis(self, boolarray, axis=0, out=None):
        """ axis-wise boolean indexing, analogous to numpy.ndarray.compress
        """
        pos, dim = self._get_axis_info(axis)
        val = self.values.compress(boolarray, axis=pos, out=out)
        newax = self.axes[pos][boolarray]
        axes = [ax if ax.name != dim else newax for ax in self.axes]
        dima = self._constructor(val, axes)
        dima.attrs.update(self.attrs)
        return dima

    def take_axis(self, indices, axis=0, indexing=None, mode='raise', out=None, side='left'):
        """ Take values along an axis, similarly to numpy.take.
        
        It is a one-dimensional version of DimArray.take for array-like indexing
        which does not accept slice or scalar indices. It may be faster
        due to less checking, and includes numpy's `mode` parameter.

        Parameters
        ----------
        indices : array-like
            Same as numpy.take except that labels can be provided. Must be iterable.
        axis : int or str, optional (default to 0)
        indexing : {'label', 'position'}, optional
            Indexing mode. 
            - "label": indexing on axis labels (default)
            - "position": use numpy-like position index
            Default value can be changed in dimarray.rcParams['indexing.by']
        mode : {"raise", "wrap", "clip"}, optional
            Specifies how out-of-bounds indices will behave.
            If `indexing=="position"`, same behaviour as numpy.take.
            If `indexing=="label", only "raise" and "clip" are allowed. 
            If `mode == 'clip'`, any label not present in the axis is clipped to 
            the nearest end of the array, or inserted at an appropriate place in the array. 
            For a sorted array, an integer position will be returned that maintained the array 
            sorted. 
            If mode == 'raise' (the default), a check is performed on the result to ensure that
            all values were present, and raise an IndexError exception otherwise.
        out : np.ndarray, optional
            Store the result (same as numpy.take)
        side : passed along to searchsorted, useful in 'clip' mode, if indexing='label'

        Returns
        -------
        dima : DimArray
            Sampled dimarray with unchanged dimensions (but different size / shape).

        Notes
        -----
        As a result of using numpy.searchsorted for array-like labels, which 
        naturally works in "clip" mode in the sense described above, it is 
        slightly faster to indicate mode == "clip" than mode == "raise" 
        (since one check less is performed)

        Examples
        --------
        >>> a = DimArray([[1,2,3],[4,5,6]], axes=[('dim0',[10.,20.]), ('dim1',['a','b','c'])])
        >>> a.take_axis([20,30,-10], axis='dim0', mode='clip')
        dimarray: 9 non-null elements (0 null)
        0 / dim0 (3): 20.0 to 10.0
        1 / dim1 (3): 'a' to 'c'
        array([[4, 5, 6],
               [4, 5, 6],
               [1, 2, 3]])
        >>> a.take_axis(['b','e'], axis='dim1', mode='clip')
        dimarray: 4 non-null elements (0 null)
        0 / dim0 (2): 10.0 to 20.0
        1 / dim1 (2): 'b' to 'c'
        array([[2, 3],
               [5, 6]])
        >>> a.dim1 = ['c','a','f'] # change axis dim1
        >>> a.take_axis(['b','e'], axis='dim1', mode='clip')
        dimarray: 4 non-null elements (0 null)
        0 / dim0 (2): 10.0 to 20.0
        1 / dim1 (2): 'c' to 'f'
        array([[1, 3],
               [4, 6]])
        """
        indexing = indexing or getattr(self, "_indexing", None) or get_option("indexing.by")

        pos, dim = self._get_axis_info(axis)
        ax = self.axes[pos]

        if not np.iterable(indices):
            raise TypeError("indices must be iterable")

        if indexing == "label":
            indices = ax.loc(indices, mode=mode)

        values = self.values.take(indices, axis=pos, mode=mode, out=out)

        axes = self.axes.copy()
        newax = ax.take(indices, mode=mode)
        newaxes = [axx.copy() if axx.name!=ax.name else newax for axx in axes]

        dima = self._constructor(values, newaxes)
        dima.attrs.update(self.attrs)

        return dima

    def _getvalues_broadcast(self, indices):
        return self.values[indices] # the default for a numpy array

    def _setvalues_broadcast(self, indices, newvalues, cast=False):
        if cast:
            self._values = _maybe_cast_type(self._values, newvalues)
        self.values[indices] = newvalues # the default for a numpy array

    def _setvalues_bool(self, mask, newvalues, cast=False):
        mask = np.asarray(mask)
        if cast:
            self._values = _maybe_cast_type(self._values, newvalues)
        self.values[mask] = newvalues # the default for a numpy array

    def _getvalues_ortho(self, indices):
        ix = orthogonal_indexer(indices, self.shape)
        return self.values[ix]

    def _setvalues_ortho(self, indices, newvalues, cast=False):
        if cast:
            self._values = _maybe_cast_type(self._values, newvalues)
        ix = orthogonal_indexer(indices, self.shape)
        self.values[ix] = newvalues
        return 

    _getaxes_broadcast = getaxes_broadcast

    def fill(self, val):
        """ anologous to numpy's fill (in-place operation)
        """
        self.values.fill(val) 

    # 
    # indexing where default behaviour is not to broadcast array indices, similar to matlab
    #
    @property
    def box(self):
        """ property to allow indexing without array broadcasting (matlab-like)
        """
        warnings.warn("box (or orthogonal) indexing has become the default indexing method of a DimArray. This method is thus deprecated.", FutureWarning)
        return self

    #
    # TRANSFORMS
    # 

    #
    # ELEMENT-WISE TRANSFORMS
    #
    def apply(self, func, *args, **kwargs):
        """ Apply element-wise function to DimArray

        Examples
        --------
        >>> DimArray([1.29, 3.11]).apply(np.round, 1)
        dimarray: 2 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        array([ 1.3,  3.1])

        >>> DimArray([-1.3, -3.1]).apply(np.abs)
        dimarray: 2 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        array([ 1.3,  3.1])
        """
        return self._constructor(func(self.values, *args, **kwargs), self.axes.copy())

    #
    # NUMPY TRANSFORMS
    #
    median = _transform.median
    sum = _transform.sum
    prod = _transform.prod

    # use weighted mean/std/var by default
    _get_weights = _transform._get_weights
    mean = _transform.mean
    std = _transform.std
    var = _transform.var

    # but provide standard mean just in case
    _mean = _transform._mean
    _std = _transform._std
    _var = _transform._var

    #_mean = _transform.mean
    #_var = _transform.var
    #_std = _transform.std

    all = _transform.all
    any = _transform.any

    min = _transform.min
    max = _transform.max
    ptp = _transform.ptp

    #
    # change `arg` to `loc`, suggesting change in indexing behaviour
    #
    argmin = _transform.argmin
    argmax = _transform.argmax

    cumsum = _transform.cumsum
    cumprod = _transform.cumprod
    diff = _transform.diff

    #
    # METHODS TO CHANGE ARRAY SHAPE AND SIZE
    #
    repeat = _reshape.repeat
    newaxis = _reshape.newaxis
    squeeze = _reshape.squeeze
    flatten = _reshape.flatten
    reshape = _reshape.reshape
    transpose = _reshape.transpose
    rollaxis = _reshape.rollaxis
    broadcast = _reshape.broadcast
    swapaxes = _reshape.swapaxes
    flatten = _reshape.flatten
    unflatten = _reshape.unflatten
    # deprecated
    group = _reshape.group
    ungroup = _reshape.ungroup
    
    @property
    def T(self):
        return self.transpose()

    #
    # REINDEXING 
    #
    reindex_axis = _align.reindex_axis
    reindex_like = _align.reindex_like
    sort_axis = _align.sort_axis

    #
    # Interpolation
    #
    interp_axis = _transform.interp_axis
    interp_like = _transform.interp_like

    # Drop missing values
    dropna = missingvalues.dropna
    fillna = missingvalues.fillna
    setna = missingvalues.setna

    # BASIC OPERATTIONS
    #
    def _binary_op(self, func, other):
        """ make an operation: this include axis and dimensions alignment

        Just for testing:
        >>> b = DimArray([[0.,1],[1,2]])
        >>> b
        ... # doctest: +SKIP
        array([[ 0.,  1.],
               [ 1.,  2.]])
        >>> np.all(b == b)
        True
        >>> np.all(b+2 == b + np.ones(b.shape)*2)
        True
        >>> np.all(b+b == b*2)
        True
        >>> np.all(b*b == b**2)
        True
        >>> np.all((b - b.values) == b - b)
        True
        >>> -b
        dimarray: 4 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        1 / x1 (2): 0 to 1
        array([[-0., -1.],
               [-1., -2.]])
        >>> np.all(-b == 0. - b)
        True

        True divide by default
        >>> a = DimArray([1,2,3])
        >>> a/2
        dimarray: 3 non-null elements (0 null)
        0 / x0 (3): 0 to 2
        array([ 0.5,  1. ,  1.5])
        >>> a//2
        dimarray: 3 non-null elements (0 null)
        0 / x0 (3): 0 to 2
        array([0, 1, 1])

        Test group/corps structure (result of operation remains DimArray)
        >>> a = DimArray([[1.,2,3],[4,5,6]])
        >>> isinstance(a + 2., DimArray)
        True
        >>> isinstance(2. + a, DimArray)
        True
        >>> isinstance(2 * a, DimArray)
        True
        >>> isinstance(a * 2, DimArray)
        True
        >>> isinstance(2 / a, DimArray)
        True
        >>> isinstance(a / 2, DimArray)
        True
        >>> isinstance(2 - a, DimArray)
        True
        >>> isinstance(a - 2, DimArray)
        True
        >>> s = 0.
        >>> for i in range(5):
        ...        s = s + a
        >>> isinstance(a, DimArray)
        True
        >>> np.all(s == 5*a)
        True

        # invert
        >>> a = DimArray([True, False])
        >>> ~a
        dimarray: 2 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        array([False,  True], dtype=bool)
        """
        result = _operation.operation(func, self, other, broadcast=get_option('op.broadcast'), reindex=get_option('op.reindex'), constructor=self._constructor)
        return result

    def _rbinary_op(self, func, other):
        return _operation.operation(func, other, self, broadcast=get_option('op.broadcast'), reindex=get_option('op.reindex'), constructor=self._constructor)

    def _unary_op(self, func):
        return self._constructor(func(self.values), self.axes)

    def __float__(self):  return float(self.values)
    def __int__(self):  return int(self.values)

    def _to_array_equiv(self, other):
        """ convert to equivalent numpy array
        raise Exception is other is a DimArray with different axes
        """
        if isinstance(other, DimArray):
            # check if axes are equal
            if self.axes != other.axes:
                raise ValueError("axes differ !")

        return np.asarray(other)

    def __eq__(self, other): 
        """ equality in numpy sense

        >>> test = DimArray([1, 2]) == 1
        >>> test
        dimarray: 2 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        array([ True, False], dtype=bool)
        >>> test2 = DimArray([1, 2]) == DimArray([1, 1])
        >>> test3 = DimArray([1, 2]) == np.array([1, 1])
        >>> np.all(test2 == test3) and np.all(test == test3)
        True
        >>> DimArray([1, 2]) == DimArray([1, 2],dims=('x1',))
        False
        """
        # check axes and convert to arrays
        try: 
            other = self._to_array_equiv(other)
        except ValueError:
            return False

        res = self.values == other

        if isinstance(res, np.ndarray):
            return self._constructor(res, self.axes)

        # boolean False
        else:
            return res

    def __ne__(self, other): 
        """ non equal 
        >>> DimArray([1, 2]) != 1
        dimarray: 2 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        array([False,  True], dtype=bool)
        """
        eq = self == other
        if isinstance(eq, bool):
            return not eq
        else:
            return ~eq

    def _cmp(self, op, other):
        """ Element-wise comparison operator

        Examples
        --------
        >>> a = DimArray([1, 2])
        >>> a < 2
        dimarray: 2 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        array([ True, False], dtype=bool)

        >>> b = DimArray([1, 2], dims=('x1',))
        >>> try: 
        ...        a > b 
        ... except ValueError, msg:
        ...        print msg        
        axes differ !
        """
        other = self._to_array_equiv(other)
        fop = getattr(self.values, op)
        return self._constructor(fop(other), self.axes)

    # comparison
    def __lt__(self, other): return self._cmp('__lt__', other)
    def __le__(self, other): return self._cmp('__le__', other)
    def __gt__(self, other): return self._cmp('__gt__', other)
    def __ge__(self, other): return self._cmp('__ge__', other)
    def __and__(self, other): return self._cmp('__and__', other)
    def __or__(self, other): return self._cmp('__or__', other)


    def __nonzero__(self):
        """ Boolean value of the object

        Examples
        --------
        >>> a = DimArray([1, 2])
        >>> try:
        ...        if DimArray([1, 2]):
        ...            print 'this does not make sense'
        ... except Exception, msg:
        ...        print msg
        The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        """
        return self.values.__nonzero__()

    __bool__ = __nonzero__

    def __contains__(self, value):
        """ Membership tests using in (needed with __nonzero__)
        """
        #return np.all(self == other)
        return self.values.__contains__(value)

    @classmethod
    def from_pandas(cls, data, dims=None):
        """ Initialize a DimArray from pandas

        Parameters
        ----------
        data : pandas object (Series, DataFrame, Panel, Panel4D)
        dims, optional : dimension (axis) names, otherwise look at ax.name for ax in data.axes

        Returns
        -------
        a : DimArray instance

        Examples
        --------

        >>> import pandas as pd
        >>> s = pd.Series([3,5,6], index=['a','b','c'])
        >>> s.index.name = 'dim0'
        >>> DimArray.from_pandas(s)
        dimarray: 3 non-null elements (0 null)
        0 / dim0 (3): 'a' to 'c'
        array([3, 5, 6])

        Also work with Multi-Index

        >>> panel = pd.Panel(np.arange(2*3*4).reshape(2,3,4))
        >>> b = panel.to_frame() # pandas' method to convert Panel to DataFrame via MultiIndex
        >>> DimArray.from_pandas(b)    # doctest: +SKIP
        dimarray: 24 non-null elements (0 null)
        0 / major,minor (12): (0, 0) to (2, 3)
        1 / x1 (2): 0 to 1
        ...  
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas module is required to use this method")

        axisnames = []
        axes = []
        for i, ax in enumerate(data.axes):
            
            # axis name
            name = ax.name
            if dims is not None: name = dims[i]
            if name is None: name = 'x%i'% (i)

            # Multi-Index: make a MultiAxis object
            if isinstance(ax, pd.MultiIndex):

                # level names
                names = ax.names
                for j, nm in enumerate(names): 
                    if nm is None:
                        names[j] = '%s_%i'%(name,j)

                miaxes = Axes.from_arrays(ax.levels, dims=names)
                axis = MultiAxis(*miaxes)

            # Index: Make a simple Axis
            else:
                axis = Axis(ax.values, name)

            axes.append(axis)

        #axisnames, axes = zip(*[(ax.name, ax.values) for ax in data.axes])

        return cls(data.values, axes=axes)

    #
    # export to other data types
    #
    def to_pandas(self):
        """ return the equivalent pandas object
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas module is required to use this method")
        obj = pandas_obj(self.values, *[ax.to_pandas() for ax in self.axes])
        return obj

    def to_cube(self):
        """ convert dimarray to iris cube
        """
        from dimarray.convert.iris import as_cube
        return as_cube(self, copy=True)

    @classmethod
    def from_cube(cls, cube):
        """ convert from iris cube
        """
        from dimarray.convert.iris import as_dimarray
        return as_dimarray(cube, copy=True, cls=cls)

#    def to_frame(self, col=0):
#        """ to pandas dataFrame
#
#        col, optional: axis to use as columns, default is 0 
#            All other dimensions are collapsed into a MultiIndex as index
#
#        Examples:
#        --------
#        ## >>> a = DimArray(np.arange(2*3*4).reshape(2,3,4))
#        ## >>> b = a.to_frame()
#        ## >>> c = a.to_frame(col='x1') # choose another axis to use as column
#        """
#        from pandas import MultiIndex, DataFrame, Index
#        pos, name = self._get_axis_info(col)
#        dims = [ax.name for ax in self.axes if ax.name != name] # all but the one designated for columns
#        if len(dims) > 0:
#        a = self.group(dims) # group all dimensions instead of col
#        ga = a.axes[0] # grouped axis, inserted as firt dimension
#        #index = MultiIndex.from_arrays(ga.values.T, names=[ax.name for ax in ga.axes])
#        index = MultiIndex.from_tuples(ga.values, names=[ax.name for ax in ga.axes])
#        columns = Index(a.axes[1].values, name=a.axes[1].name)
#
#        return DataFrame(a.values, index=index, columns=columns)


    # Split along an axis
    def to_odict(self, axis=0):
        """ Return a dictionary of DimArray

        .. deprecated:: 1.9
        """
        warnings.warn('yo',category=DeprecationWarning)

        d = odict()
        for k, val in self.iter(axis):
            d[k] = val
        return d

    def to_jsondict(self):
        """ return a dictionary representation of a DimArray, which is suitable 
        for conversion to json format
        """
        jsondict = odict([
            ('values', self.values.tolist()),
            ('dims', list(self.dims)),
            ('labels', [ax.values.tolist() for ax in self.axes]),
            ('shape', list(self.shape)), 
            ('ndim', self.ndim),
        ])

        # add metadata
        # only include metadata that can be converted into json format
        try:
            import json
            jsonimported=True
        except ImportError:
            jsonimported=False

        meta = {}
        for m in self._metadata():
            try:
                val = getattr(self, m)
                if jsonimported: 
                    _ = json.dumps(val)
                meta[m] = val
            except: 
                warnings.warn("could not jsonify metadata "+m)
        jsondict['meta'] = meta

        return jsondict

    def to_json(self, separators=(',',':'), **kwargs):
        """ json representation of a dimarray

        Parameters
        ----------
        separators, **kwargs : passed to json.dumps

        Returns
        -------
        json string

        Note
        ----
        shape and ndim are provided for informative purpose, but are not used
        when initializing a DimArray from a json string

        Example
        -------
        >>> a = DimArray([[1.,2.,3.],[4.,5.,6.]], axes=[[0,1],[10,11,12]], dims=['a','b'])
        >>> a.to_json()
        '{"values":[[1.0,2.0,3.0],[4.0,5.0,6.0]],"dims":["a","b"],"labels":[[0,1],[10,11,12]],"shape":[2,3],"ndim":2,"meta":{}}'
        """
        import json
        return json.dumps(self.to_jsondict(), separators=separators, **kwargs)

    @classmethod
    def from_jsondict(cls, jsondict):
        """ initialize a DimArray from a json-compatible dictionary
        """
        jsondict = jsondict.copy()
        dima = cls(jsondict.pop('values', None), 
                   axes=jsondict.pop('labels', None), 
                   dims=jsondict.pop('dims', None))
        if 'meta' in jsondict:
            dima._metadata(jsondict['meta'])
        return dima

    @classmethod
    def from_json(cls, s):
        """ return a DimArray, from a json string

        >>> s = '{"values":[[1.0,2.0,3.0],[4.0,5.0,6.0]],"dims":["a","b"],"labels":[[0,1],[10,11,12]],"shape":[2,3],"ndim":2,"meta":{}}'
        >>> DimArray.from_json(s)
        dimarray: 6 non-null elements (0 null)
        0 / a (2): 0 to 1
        1 / b (3): 10 to 12
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]])
        """
        import json
        jsondict = json.loads(s)
        return cls.from_jsondict(jsondict)

    def to_dataset(self, axis=0):
        """ split a DimArray into a Dataset object (collection of DimArrays)
        """
        from dimarray.dataset import Dataset
        # iterate over elements of one axis
        #data = [val for k, val in self.iter(axis)]
        # Dataset(data, keys=self.axes[axis].values)
        ds =  Dataset()
        for k, val in self.iter(axis):
            if not isinstance(val, DimArray): # scalar case
                val = DimArray(val)
            ds[k] = val
        return ds

    def to_MaskedArray(self, copy=True):
        """ transform to MaskedArray, with NaNs as missing values
        """
        values = self.values
        if anynan(values):
            mask = np.isnan(values)
        else:
            mask = False
        values = np.ma.array(values, mask=mask, copy=copy)
        return values

    def to_list(self, axis=0):
        return [val for  k, val in self.iter(axis)]

    def to_larry(self):
        """ return the equivalent pandas object
        """
        try:
            import la
        except ImportError:
            raise ImportError("la (larry) module is required to use this method")

        a = la.larry(self.values, [list(ax.values) for ax in self.axes])
        return a

    #
    # pretty printing
    #
    _repr = repr_dimarray

    #
    #  I/O
    # 
    def write_nc(self, f, name=None, mode='w', clobber=None, format=None, *args, **kwargs):
        """ Write to netCDF

        Parameters
        ----------
        f : file name
        name : variable name, optional
            must be provided if no attribute "name" is defined
        mode, clobber, format : see netCDF4.Dataset
        **kwargs : passed to netCDF4.Dataset.createVAriable (compression)

        See Also
        --------
        DatasetOnDisk
        """
        import dimarray.io.nc as ncio
        f, close = ncio._maybe_open_file(f, mode=mode, clobber=clobber, format=format)
        store = ncio.DatasetOnDisk(f, mode=mode, clobber=clobber)
        store.write(name, self, *args, **kwargs)
        if close: store.close()

    # Aliases
    write = write_nc

    #
    # Plotting
    #
    _plot2D = plotting._plot2D
    _plot1D = plotting._plot1D
    plot = plotting.plot
    bar = plotting.bar
    barh = plotting.barh
    stackplot = plotting.stackplot
    contourf = plotting.contourf
    contour = plotting.contour
    pcolor = plotting.pcolor

    def set_axis(self, values=None, axis=0, name=None, inplace=True, **kwargs):
        """ Set axis values, name and attributes
        
        Parameters
        ----------
        values : numpy array-like or mapper (callable or dict), optional
            - array-like : new axis values, must have exactly the same 
            length as original axis
            - dict : establish a map between original and new axis values
            - callable : transform each axis value into a new one
            - if None, axis values are left unchanged
            Default to None.
        axis : int or str, optional
            axis to be (re)set
        name : str, optional
            rename axis
        inplace : bool, optional
            modify dataset axis in-place (True) or return copy (False)? 
            (default True)
        **kwargs : key-word arguments
            Also reset other axis attributes, which can be single metadata
            or other axis attributes, via using `setattr`
            This includes special attributes `weights` and `attrs` (the latter
            reset all attributes)

        Returns
        -------
        None, or DimArray instance if inplace is False

        Examples
        --------
        >>> a = DimArray([1, 2, 3, 4], axes = [[ 1900, 1901, 1902, 1903 ]], dims=['time'])
        >>> a
        dimarray: 4 non-null elements (0 null)
        0 / time (4): 1900 to 1903
        array([1, 2, 3, 4])

        Provide new values for the whole axis

        >>> a.set_axis(list('abcd'), inplace=False)
        dimarray: 4 non-null elements (0 null)
        0 / time (4): 'a' to 'd'
        array([1, 2, 3, 4])

        Update just a few of the values

        >>> a.set_axis({1900:-9999}, inplace=False)
        dimarray: 4 non-null elements (0 null)
        0 / time (4): -9999 to 1903
        array([1, 2, 3, 4])

        Or transform axis values

        >>> a.set_axis(lambda x: x*0.01, inplace=False)
        dimarray: 4 non-null elements (0 null)
        0 / time (4): 19.0 to 19.03
        array([1, 2, 3, 4])

        Only change name. 

        >>> a.set_axis(name='year', inplace=False)
        dimarray: 4 non-null elements (0 null)
        0 / year (4): 1900 to 1903
        array([1, 2, 3, 4])
         
        This is the equivalent of

        >>> a.axes['time'].name = 'year'
        >>> a
        dimarray: 4 non-null elements (0 null)
        0 / year (4): 1900 to 1903
        array([1, 2, 3, 4])
        """
        if not inplace: self = self.copy()
        self.axes[axis].set(values=values, inplace=True, name=name, **kwargs)
        if not inplace: return self

    def reset_axis(values=None, axis=0, **kwargs):
        warnings.warn(FutureWarning('reset_axis is deprecated, use set_axis'))
        if values is None: values = np.arange(self.axes[axis], size)
        if values is False: values = None
        return self.set(values, name, axis=axis, **kwargs)


    # for back-compatibility
    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """ use dimarray.array instead """
        # kwargs['cls'] = cls
        warnings.warn(FutureWarning('from_arrays is deprecated, use DimArray() or concatenate() or stack() or Dataset().to_array() instead'))
        return cls(array(*args, **kwargs))

# back compat
from_arrays = DimArray.from_arrays

def array(data, *args, **kwargs):
    """ initialize a DimArray by providing axes as key-word arguments

    Parameters
    ----------
    data: numpy array-like 
        list or dict of DimArray objects ==> call stack(data, *args, **kwargs)

    *args: variable-list arguments
    **kwargs: key-word arguments

    Returns
    -------
    DimArray

    if data is an array-like, this call is the same as DimArray(data, *args, **kwargs)
    if data is a dict, list or tuple, this call is similar to da.stack(data, *args, **kwargs) but 
        with align=True by default and an additional parameter broadcast=True by default
        This behaviour might change.

    See also
    --------
    DimArray, stack, Dataset.to_array

    Examples
    --------

    From a list:

    >>> import dimarray as da
    >>> a = DimArray([1,2,3])
    >>> da.array([a, 2*a]) # if keys not provided, default is 0, 1
    dimarray: 6 non-null elements (0 null)
    0 / unnamed (2): 0 to 1
    1 / x0 (3): 0 to 2
    array([[1, 2, 3],
           [2, 4, 6]])

    From a dict, here also with axis alignment, and naming the axis

    >>> d = {'a':DimArray([10,20,30.],[0,1,2]), 'b':DimArray([1,2,3.],[1.,2.,3])}
    >>> a = da.array(d, keys=['a','b'], axis='items') # keys= just needed to enforce ordering
    >>> a
    dimarray: 6 non-null elements (2 null)
    0 / items (2): 'a' to 'b'
    1 / x0 (4): 0.0 to 3.0
    array([[ 10.,  20.,  30.,  nan],
           [ nan,   1.,   2.,   3.]])

    Concatenate 2-D data

    >>> a = DimArray([[0,1],[2,3.]])
    >>> b = a.copy()
    >>> b[0,0] = np.nan
    >>> c = da.array([a,b],keys=['a','b'],axis='items')
    >>> d = da.array({'a':a,'b':b},axis='items')
    >>> np.all(np.isnan(c) | (c == d) )
    True
    >>> c
    dimarray: 7 non-null elements (1 null)
    0 / items (2): 'a' to 'b'
    1 / x0 (2): 0 to 1
    2 / x1 (2): 0 to 1
    array([[[  0.,   1.],
            [  2.,   3.]],
    <BLANKLINE>
           [[ nan,   1.],
            [  2.,   3.]]])
    """
    # if some kind of dictionary, first transform to list of values and keys
    if isinstance(data, dict):
        from collections import OrderedDict as odict

        d = odict()
        keys = kwargs.pop('keys', data.keys())
        for k in keys:
            d[k] = data[k]

        data = d.values()
        kwargs['keys'] = keys

    # sequence: align axes and stack arrays
    if (isinstance(data, list) or isinstance(data, tuple)) and len(data) > 0 and isinstance(data[0], DimArray):

        # reindex and broad-cast arrays? (by default, yes)
        reindex = kwargs.pop('align', True)
        broadcast = kwargs.pop('broadcast', True)

        if reindex:
            data = align(data)

        if broadcast:
            data = broadcast_arrays(*data) # make sure the arrays have the same dimension

        kwargs['align'] = False # already aligned

        return stack(data, *args, **kwargs) 

    # equivalent to calling DimArray
    else:
        return DimArray(data, *args, **kwargs)

# HANDY ALIAS
from_pandas = DimArray.from_pandas

def empty(axes=None, dims=None, shape=None, dtype=float):
    """ Initialize an empty array

    axes and dims have the same meaning as DimArray's initializer
    shape, optional: can be provided in combination with `dims`
        if `axes=` is omitted.

    >>> a = empty([('time',[2000,2001]),('items',['a','b','c'])])
    >>> a.fill(3)
    >>> a
    dimarray: 6 non-null elements (0 null)
    0 / time (2): 2000 to 2001
    1 / items (3): 'a' to 'c'
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]])
    >>> b = empty(dims=('time','items'), shape=(2, 3))

    See also
    --------
    empty_like, ones, zeros, nans
    """
    axes = Axes._init(axes, dims=dims, shape=shape)

    if shape is None:
        shape = [ax.size for ax in axes]

    values = np.empty(shape, dtype=dtype)

    return DimArray(values, axes=axes)

def empty_like(a, dtype=None):
    """ alias for empty(a.axes, dtype=a.dtype)

    See also
    --------
    empty, ones_like, zeros_like, nans_like

    >>> a = empty([('time',[2000,2001]),('items',['a','b','c'])])
    >>> b = empty_like(a)
    >>> b.fill(3)
    >>> b
    dimarray: 6 non-null elements (0 null)
    0 / time (2): 2000 to 2001
    1 / items (3): 'a' to 'c'
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]])
    """
    if dtype is None: dtype = a.dtype
    return empty(a.axes, dtype=dtype)

def nans(axes=None, dims=None, shape=None):
    """ Initialize an empty array filled with NaNs. See empty for doc.

    >>> nans(dims=('time','items'), shape=(2, 3))
    dimarray: 0 non-null elements (6 null)
    0 / time (2): 0 to 1
    1 / items (3): 0 to 2
    array([[ nan,  nan,  nan],
           [ nan,  nan,  nan]])
    """
    a = empty(axes, dims, shape, dtype=float)
    a.fill(np.nan)
    return a

def nans_like(a):
    """ alias for nans(a.axes, dtype=a.dtype)
    """
    return nans(a.axes)

def ones(axes=None, dims=None, shape=None, dtype=float):
    """ Initialize an empty array filled with ones. See empty for doc.

    >>> ones(dims=('time','items'), shape=(2, 3))
    dimarray: 6 non-null elements (0 null)
    0 / time (2): 0 to 1
    1 / items (3): 0 to 2
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])
    """
    a = empty(axes, dims, shape, dtype=dtype)
    a.fill(1)
    return a

def ones_like(a, dtype=None):
    """ alias for nans(a.axes, dtype=a.dtype)
    """
    if dtype is None: dtype = a.dtype
    return ones(a.axes, dtype=dtype)

def zeros(axes=None, dims=None, shape=None, dtype=float):
    """ Initialize an array filled with zeros. See empty for doc.

    >>> zeros(dims=('time','items'), shape=(2, 3))
    dimarray: 6 non-null elements (0 null)
    0 / time (2): 0 to 1
    1 / items (3): 0 to 2
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])
    """
    a = empty(axes, dims, shape, dtype=dtype)
    a.fill(0)
    return a

def zeros_like(a, dtype=None):
    """ alias for zeros(a.axes, dtype=a.dtype)
    """
    if dtype is None: dtype = a.dtype
    return zeros(a.axes, dtype=dtype)


def _is_dictlike(dict_):
    return hasattr(dict_, 'keys') and hasattr(dict_, '__getitem__') and hasattr(dict_,'values')

def _contains_dictlike(dict_):
    """ true if a dict or dataframe in dict
    """
    return _is_dictlike(dict_) or isinstance(dict_, list) \
            and np.any([_contains_dictlike(item) for item in dict_])

