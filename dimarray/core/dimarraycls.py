#encoding:utf-8

""" array with physical dimensions (named and valued axes)
"""
import numpy as np
import copy
import warnings
from collections import OrderedDict as odict

from dimarray.config import get_option

from metadata import MetadataDesc
from axes import Axis, Axes, GroupedAxis, _doc_reset_axis

import transform as _transform  # numpy along-axis transformations, interpolation
import reshape as _reshape      # change array shape and dimensions
import indexing as _indexing    # perform slicing and indexing operations
import operation as _operation  # operation between DimArrays
import missingvalues # operation between DimArrays
from align import broadcast_arrays, align_axes, stack

from tools import pandas_obj

__all__ = ["DimArray", "array"]

class DimArray(object):
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
    broadcast : repeat array to match target dimensions
    reindex_like : re-index like another dimarray / axes instance
    reindex_axis : re-index / interpolate an axis
    reset_axis : reset axis / change axis values & metadata

    reshape 
    group : flatten particular sets of dimensions
    ungroup : inflate back
    swapaxes : swap two axes, e.g. a.swapaxes(0, 'x1') 
    transpose, squeeze : similar to ndarray's methods

    to_pandas : convert to pandas object (must be <= 4D) (pretty printing)
    from_pandas : convert from pandas object, support MultiIndex

    to_dataset : split the dimarray along an axis to create a Dataset 

    write_nc : write DimArray to netCDF file

    plot
    pcolor
    contourf
    contour

    Notes
    -----
    see interactive help for a full listing of methods with doc

    See Also
    --------
    Axes, Axis, GroupedAxis, Dataset
    read_nc, stack, concatenate
    array, array_kw

    Examples
    --------

    >>> a = DimArray([[1.,2,3], [4,5,6]], axes=[['grl', 'ant'], [1950, 1960, 1970]], dims=['variable', 'time']) 
    >>> a
    dimarray: 6 non-null elements (0 null)
    dimensions: 'variable', 'time'
    0 / variable (2): grl to ant
    1 / time (3): 1950 to 1970
    array([[ 1.,  2.,  3.],
	   [ 4.,  5.,  6.]])

    Array data are stored in a `values` attribute:

    >>> a.values
    array([[ 1.,  2.,  3.],
	   [ 4.,  5.,  6.]])

    while its axes are stored in `axes`:

    >>> a.axes
    dimensions: 'variable', 'time'
    0 / variable (2): grl to ant
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
    dimensions: 'variable'
    0 / variable (2): grl to ant
    array([ 2.,  5.])

    During an operation, arrays are automatically re-indexed to span the 
    same axis domain, with nan filling if needed

    >>> a = DimArray([0, 1], axes = [[0, 1]])
    >>> b = DimArray([0,1,2], axes = [[0, 1, 2]])
    >>> a+b
    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  0.,   2.,  nan])

    Dimensions are factored (broadcast) when performing an operation

    >>> a = DimArray([0, 1], dims=('x0',))
    >>> b = DimArray([0, 1, 2], dims=('x1',))
    >>> a+b
    dimarray: 6 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[0, 1, 2],
	   [1, 2, 3]])
    """
    _order = None  # set a general ordering relationship for dimensions
    _metadata = MetadataDesc(exclude=('values','axes'))

    #
    # NOW MAIN BODY OF THE CLASS
    #

    def __init__(self, values=None, axes=None, dims=None, labels=None, copy=False, dtype=None, _indexing=None, _indexing_broadcast=None, **kwargs):
        """ Initialize a DimArray instance

	Parameters
	----------

	values : numpy-like array, or DimArray instance
		  If `values` is not provided, will initialize an empty array 
		  with dimensions inferred from axes (in that case `axes=` 
		  must be provided).

	axes : optional, list or tuple 
	
	    axis values as ndarrays, whose order 
	    matches axis names (the dimensions) provided via `dims=` 
	    parameter. Each axis can also be provided as a tuple 
	    (str, array-like) which contains both axis name and axis 
	    values, in which case `dims=` becomes superfluous.
	    `axes=` can also be provided with a list of Axis objects
	    If `axes=` is omitted, a standard axis `np.arange(shape[i])`
	    is created for each axis `i`.

	dims : optional, list or tuple
	    dimensions (or axis names)
	    This parameter can be omitted if dimensions are already 
	    provided by other means, such as passing a list of tuple 
	    to `axes=`. If axes are passed as keyword arguments (via 
	    **kwargs), `dims=` is used to determine the order of 
	    dimensions. If `dims` is not provided by any of the means 
	    mentioned above, default dimension names are 
	    given `x0`, `x1`, ...`xn`, where n is the number of 
	    dimensions.

	dtype : optional, data type, passed to np.array() 
	copy : optional, passed to np.array()

	**kwargs : metadata 
	
	    NOTE: metadata passed this way cannot have name already taken by other 
	    parameters such as "values", "axes", "dims", "dtype" or "copy".

	Examples
	--------

	Basic:

	>>> DimArray([[1,2,3],[4,5,6]]) # automatic labelling
	dimarray: 6 non-null elements (0 null)
	dimensions: 'x0', 'x1'
	0 / x0 (2): 0 to 1
	1 / x1 (3): 0 to 2
	array([[1, 2, 3],
	       [4, 5, 6]])

	>>> DimArray([[1,2,3],[4,5,6]], dims=['items','time'])  # axis names only
	dimarray: 6 non-null elements (0 null)
	dimensions: 'items', 'time'
	0 / items (2): 0 to 1
	1 / time (3): 0 to 2
	array([[1, 2, 3],
	       [4, 5, 6]])

	>>> DimArray([[1,2,3],[4,5,6]], axes=[list("ab"), np.arange(1950,1953)]) # axis values only
	dimarray: 6 non-null elements (0 null)
	dimensions: 'x0', 'x1'
	0 / x0 (2): a to b
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
	dimensions: 'items', 'time'
	0 / items (2): a to b
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

        if values is not None:
            values = np.array(values, copy=copy, dtype=dtype)

        #
        # Initialize the axes
        # 
        if axes is None and labels is None:
            assert values is not None, "values= or/and axes=, labels= required to determine dimensions"

        axes = Axes._init(axes, dims=dims, labels=labels, shape=values.shape if values is not None else None)
        assert type(axes) is Axes

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
        self.values = values
        self.axes = axes

        ## options
        self._indexing = _indexing
        self._indexing_broadcast = _indexing_broadcast

        #
        # metadata (see Metadata type in metadata.py)
        #
        #for k in kwargs:
        #    setncattr(self, k, kwargs[k]) # perform type-checking and store in self._metadata
        self._metadata = kwargs

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
            self.values = obj.values
            self.axes = obj.axes

    @classmethod
    def from_kw(cls, *args, **kwargs):
        """ See array_kw for the doc
        """
        if len(args) == 0:
            values = None
        else:
            values = args[0]

        if len(args) > 1:
            dims = args[1]
        else:
            dims = None

        axes = {k:kwargs[k] for k in kwargs}

        assert len(args) <= 2, "[values, [dims]]"

        return cls(values, axes, dims)

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
        #if '_indexing' in metadata:
        #    kwargs['_indexing'] = metadata.pop('_indexing')
        #obj = cls(values, axes)
        #for k in metadata: obj.setncattr(k, metadata[k])
        return cls(values, axes, **metadata)

    #
    # Attributes access
    #
    def __getattr__(self, att):
        """ allows for accessing axis values by '.' directly
        """
        # exclude special attributes from checking
        exclude = ['values', 'axes']

        # check for dimensions
        if not att.startswith('_') and att not in exclude and att in self.dims:
            ax = self.axes[att]
            return ax.values # return numpy array

        else:
            raise AttributeError("{} object has no attribute {}".format(self.__class__.__name__, att))

    def copy(self, shallow=False):
        """ copy of the object and update arguments

        shallow: if True, does not copy values and axes
        """
        import copy
        new = copy.copy(self) # shallow copy

        if not shallow:
            new.values = self.values.copy()
            new.axes = self.axes.copy()

        return new
        #return DimArray(self.values.copy(), self.axes.copy(), slicing=self.slicing, **{k:getattr(self,k) for k in self.ncattrs()})

    #
    # Basic numpy array's properties
    #
    # basic numpy shape attributes as properties
    @property
    def shape(self): 
        return self.values.shape

    @property
    def size(self): 
        return self.values.size

    @property
    def ndim(self): 
        return self.values.ndim

    #
    # Some additional properties
    #
    @property
    def dims(self):
        """ axis names 
        """
        return tuple([ax.name for ax in self.axes])

    @property
    def labels(self):
        """ axis values 
        """
        return tuple([ax.values for ax in self.axes])

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
        dimensions: 'xx'
        0 / xx (2): a to b
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

    def iter(self, axis=0):
        """ Iterate over axis value and cross-section, along any axis (by default the first)

        for time, time_slice in myarray.iter('time'):
            do stuff
        """
        # iterate over axis values
        for k in self.axes[axis].values:
            val = self.take(k, axis=axis) # cross-section
            yield k, val

    #
    # returns axis position and name based on either of them
    #
    def _get_axis_info(self, axis):
        """ axis position and name

        Parameters
        ----------
            axis: `int` or `str` or None

        Returns
        -------
            idx        : `int`, axis position
            name: `str` or None, axis name
        """
        if axis is None:
            return None, None

        if type(axis) is str:
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
            axes: sequence of str or int, representing axis (dimension) 
                names or positions, possibly mixed up.

        Returns
        -------
            pos          : list of `int` indicating dimension's rank in the array
            names : list of dimension names
        """
        pos, names = zip(*[self._get_axis_info(x) for x in axes])
        return pos, names

    #
    # Return a weight for each array element, used for `mean`, `std` and `var`
    #
    def get_weights(self, weights=None, axis=None, fill_nans=True):
        """ Weights associated to each array element
        
        Parameters
        ----------
            weights: (numpy)array-like of weights, taken from axes if None or "axis" or "axes"
            axis   : int or str, if None a N-D weight is created

        Returns
        -------
            weights: DimArray of weights, or None

        Notes
        -----
        this function is used by `mean`, `std` and `var`
        """
        # make sure weights is a nd-array
        if weights is not None:
            weights = np.array(weights) 

        if weights in ('axis','axes'):
            weights = None

        if axis is None:
            dims = self.dims

        elif type(axis) is tuple:
            dims = axis

        else:
            dims = (axis,)

        # axes over which the weight is defined
        axes = [ax for ax in self.axes if ax.name in dims]

        # Create weights from the axes if not provided
        if weights is None:

            weights = 1
            for axis in dims:
                ax = self.axes[axis]
                if ax.weights is None: continue
                weights = ax.get_weights().reshape(dims) * weights

            if weights is 1:
                return None
            else:
                weights = weights.values

        # Now create a dimarray and make it full size
        # ...already full-size
        if weights.shape == self.shape:
            weights = DimArray(weights, self.axes)

        # ...need to be expanded
        elif weights.shape == tuple(ax.size for ax in axes):
            weights = DimArray(weights, axes).expand(self.dims)

        else:
            try:
                weights = weights + np.zeros_like(self.values)

            except:
                raise ValueError("weight dimensions are not conform")
            weights = DimArray(weights, self.axes)

        #else:
        #    raise ValueError("weight dimensions not conform")

        # fill NaNs in when necessary
        if fill_nans and hasnan(self.values):
            weights.values[np.isnan(self.values)] = np.nan
        
        assert weights is not None
        return weights


    #
    # INDEXING
    #

    #
    # New general-purpose indexing method
    #
    take = _indexing.take
    put = _indexing.put  
    fill = _indexing.fill  

    # get the kw index equivalent {dimname:indices}
    _get_keyword_indices = _indexing._get_keyword_indices 

    #
    # Standard indexing takes axis values
    #
    def __getitem__(self, indices): 
        """ get a slice (use take method)
        """
        #indexing = self.axes.loc[indices]
        #return self.take(indexing, position_index=True)
        return self.take(indices, indexing=self._indexing, broadcast_arrays=self._indexing_broadcast)

    def __setitem__(self, ix, val):
        """
        """
        self.put(val, ix, indexing=self._indexing, broadcast_arrays=self._indexing_broadcast, inplace=True)

    # 
    # Can also use integer-indexing via ix
    #
    @property
    def ix(self):
        """ integer index-access (toogle between integer-based and values-based indexing)
        """
        if self._indexing is "values": 
            _indexing = "position"
        else:
            _indexing = "values"
        return self._constructor(self.values, self.axes, _indexing=_indexing, _indexing_broadcast=self._indexing_broadcast, **self._metadata)

    # 
    # indexing where default behaviour is not to broadcast array indices, similar to matlab
    #
    @property
    def box(self):
        return self._constructor(self.values, self.axes, _indexing=self._indexing , _indexing_broadcast= not self._indexing_broadcast, **self._metadata)

    @property
    def box_ix(self):
        return self.box.ix

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
        dimensions: 'x0'
        0 / x0 (2): 0 to 1
        array([ 1.3,  3.1])

        >>> DimArray([-1.3, -3.1]).apply(np.abs)
        dimarray: 2 non-null elements (0 null)
        dimensions: 'x0'
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
    _mean = _transform._mean
    _std = _transform._std
    _var = _transform._var
    mean = _transform.weighted_mean
    std = _transform.weighted_std
    var = _transform.weighted_var

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
    broadcast = _reshape.broadcast
    groupby = _reshape.groupby
    swapaxes = _reshape.swapaxes
    group = _reshape.group
    ungroup = _reshape.ungroup
    
    @property
    def T(self):
        return self.transpose()

    #
    # REINDEXING 
    #
    reindex_axis = _indexing.reindex_axis
    reindex_like = _indexing.reindex_like
    sort_axis = _indexing.sort_axis

    # Drop missing values
    dropna = missingvalues.dropna
    fillna = missingvalues.fillna
    setna = missingvalues.setna

    #
    # BASIC OPERATTIONS
    #
    def _operation(self, func, other):
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
        dimensions: 'x0', 'x1'
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
        dimensions: 'x0'
        0 / x0 (3): 0 to 2
        array([ 0.5,  1. ,  1.5])
        >>> a//2
        dimarray: 3 non-null elements (0 null)
        dimensions: 'x0'
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
        """
        result = _operation.operation(func, self, other, broadcast=get_option('op.broadcast'), reindex=get_option('op.reindex'), constructor=self._constructor)
        return result

    def _roperation(self, func, other):
        return _operation.operation(func, other, self, broadcast=get_option('op.broadcast'), reindex=get_option('op.reindex'), constructor=self._constructor)

    def __neg__(self): return self._constructor(-self.values, self.axes)
    def __pos__(self): return self._constructor(+self.values, self.axes)

    def __add__(self, other): return self._operation(np.add, other)
    def __sub__(self, other): return self._operation(np.subtract, other)
    def __mul__(self, other): return self._operation(np.multiply, other)

    def __div__(self, other): return self._operation(np.true_divide, other) # TRUE DIVIDE
    def __truediv__(self, other): return self._operation(np.true_divide, other)
    def __floordiv__(self, other): return self._operation(np.floor_divide, other)

    def __pow__(self, other): return self._operation(np.power, other)
    def __sqrt__(self, other): return self**0.5

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return self._roperation(np.subtract, other)
    def __rdiv__(self, other): return self._roperation(np.true_divide, other)
    def __rpow__(self, other): return self._roperation(np.power, other)


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
        dimensions: 'x0'
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

    def __invert__(self): 
        """
        Examples:
        >>> a = DimArray([True, False])
        >>> ~a
        dimarray: 2 non-null elements (0 null)
        dimensions: 'x0'
        0 / x0 (2): 0 to 1
        array([False,  True], dtype=bool)
        """
        return self._constructor(np.invert(self.values), self.axes)

    def _cmp(self, op, other):
        """ Element-wise comparison operator

        >>> a = DimArray([1, 2])
        >>> a < 2
        dimarray: 2 non-null elements (0 null)
        dimensions: 'x0'
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

    def __lt__(self, other): return self._cmp('__lt__', other)
    def __le__(self, other): return self._cmp('__le__', other)
    def __gt__(self, other): return self._cmp('__gt__', other)
    def __ge__(self, other): return self._cmp('__ge__', other)
    def __and__(self, other): return self._cmp('__and__', other)
    def __or__(self, other): return self._cmp('__or__', other)

    def __nonzero__(self):
        """ Boolean value of the object

        Examples:
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
        dimensions: 'dim0'
        0 / dim0 (3): a to c
        array([3, 5, 6])

        Also work with Multi-Index

        >>> panel = pd.Panel(np.arange(2*3*4).reshape(2,3,4))
        >>> b = panel.to_frame() # pandas' method to convert Panel to DataFrame via MultiIndex
        >>> DimArray.from_pandas(b)    # doctest: +SKIP
        dimarray: 24 non-null elements (0 null)
        dimensions: 'major,minor', 'x1'
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

            # Multi-Index: make a Grouped Axis object
            if isinstance(ax, pd.MultiIndex):

                # level names
                names = ax.names
                for j, nm in enumerate(names): 
                    if nm is None:
                        names[j] = '%s_%i'%(name,j)

                miaxes = Axes.from_arrays(ax.levels, dims=names)
                axis = GroupedAxis(*miaxes)

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
        """
        """
        d = odict()
        for k, val in self.iter(axis):
            d[k] = val
        return d

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

    def to_MaskedArray(self):
        """ transform to MaskedArray, with NaNs as missing values
        """
        return _transform.to_MaskedArray(self.values)

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

    def __repr__(self):
        """ pretty printing
        """
        try:
            if self.ndim > 0:
                nonnull = np.size(self.values[~np.isnan(self.values)])
            else:
                nonnull = ~np.isnan(self.values)

        except TypeError: # e.g. object
            nonnull = self.size

        lines = []

        #if self.size < 10:
        #    line = "dimarray: "+repr(self.values)
        #else:
        line = "dimarray: {} non-null elements ({} null)".format(nonnull, self.size-nonnull)
        lines.append(line)

        # # show metadata as well?
        # If len(self.ncattrs()) > 0:
        #     line = self.repr_meta()
        #     lines.append(line)

        if True: #self.size > 1:
            line = repr(self.axes)
            lines.append(line)

        if self.size < get_option('display.max'):
            line = repr(self.values)
        else:
            line = "array(...)"
        lines.append(line)

        return "\n".join(lines)


    #
    #  I/O
    # 
    def write_nc(self, f, name=None, *args, **kwargs):
        """ Write to netCDF

        If you read this documenation, it means that netCDF4 is not installed on your 
        system and the write_nc / read_nc function will raise an Exception.
        """
        import dimarray.io.nc as ncio

        # add variable name if provided...
        if name is None and hasattr(self, "name"):
            name = self.name

        ncio._write_variable(f, self, name, *args, **kwargs)

    @classmethod
    def read_nc(cls, f, *args, **kwargs):
        """ Read from netCDF

        If you read this documenation, it means that netCDF4 is not installed on your 
        system and the write_nc / read_nc function will raise an Exception.
        """
        import dimarray.io.nc as ncio
        return ncio._read_variable(f, *args, **kwargs)

    # Aliases
    write = write_nc
    read = read_nc

    #
    # Plotting
#    #
#    def scatter_vs(self, axis=0, **kwargs):
#        """ make a plot along the first dim
#        """
#        if len(self.dims) > 1:
#            raise NotImplementedError("plot only valid grouped by 1 variable")
#        import matplotlib.pyplot as plt
#        if 'ax' in kwargs:
#            ax = kwargs.pop('ax')
#        else:
#            ax = plt.gca()
#
#        return ax.plot(self.a.axes[0].values, *args, **kwargs)

    def plot(self, *args, **kwargs):
        """ by default, use pandas for plotting (for now at least)
        """
        assert self.ndim <= 2, "only support plotting for 1- and 2-D objects"
        return self.to_pandas().plot(*args, **kwargs)
    
    def _plot2D(self, function, *args, **kwargs):
        """ generic plotting function for 2-D plots
        """
        if len(self.dims) != 2:
            raise NotImplementedError("pcolor can only be called on two-dimensional dimarrays.")

        import matplotlib.pyplot as plt
        
        ax = plt.gca()
        pc = function(self.labels[1], self.labels[0], self.values, **kwargs)
        plt.xlabel(self.dims[1])
        plt.ylabel(self.dims[0])
            
        return pc

    def pcolor(self, *args, **kwargs):
        """ Plot a quadrilateral mesh. 
        
        Wraps matplotlib pcolormesh().
        See pcolormesh documentation in matplotlib for accepted keyword arguments.
        
        Examples
        --------
        x = DimArray(np.zeros([100,40]))
        x.pcolor()
        x.T.pcolor() # to flip horizontal/vertical axes
        """
        
        if len(self.dims) != 2:
            raise NotImplementedError("pcolor can only be called on two-dimensional dimarrays.")

        import matplotlib.pyplot as plt

        return self._plot2D(plt.pcolormesh, *args, **kwargs)

    def contourf(self, *args, **kwargs):
        """ Plot filled 2-D contours. 
        
        Wraps matplotlib contourf().
        See contourf documentation in matplotlib for accepted keyword arguments.
        
        Examples
        --------
        x = DimArray(np.zeros([100,40]))
        x[:50,:20] = 1.
        x.contourf()
        x.T.contourf() # to flip horizontal/vertical axes
        """
        if len(self.dims) != 2:
            raise NotImplementedError("contourf can only be called on two-dimensional dimarrays.")

        import matplotlib.pyplot as plt

        return self._plot2D(plt.contourf, *args, **kwargs)

    def contour(self, *args, **kwargs):
        """ Plot 2-D contours. 
        
        Wraps matplotlib contour().
        See contour documentation in matplotlib for accepted keyword arguments.
        
        Examples
        --------
        x = DimArray(np.zeros([100,40]))
        x[:50,:20] = 1.
        x.contour()
        x.T.contour() # to flip horizontal/vertical axes
        """
        if len(self.dims) != 2:
            raise NotImplementedError("contour can only be called on two-dimensional dimarrays.")

        import matplotlib.pyplot as plt

        return self._plot2D(plt.contour, *args, **kwargs)
        
    # reset axis values
    def reset_axis(self, values=None, axis=0, inplace=False, **kwargs):
	""" Reset axis values and attributes

	Parameters
	----------
	{values}
	{axis}
	{inplace}
	{kwargs}

	Returns
	-------
	DimArray instance, or None if inplace is True
	"""
	axes = self.axes.reset_axis(values, axis, inplace=inplace, **kwargs)

	# make a copy?
	if not inplace:
	    a = self.copy()
	    a.axes = axes
	    return a

    # for back-compatibility
    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """ use dimarray.array instead """
        kwargs['cls'] = cls
        warnings.warn(FutureWarning('from_arrays is deprecated, use da.array() instead'))
        return array(*args, **kwargs)


DimArray.reset_axis.__func__.__doc__ = DimArray.reset_axis.__func__.__doc__.format(**_doc_reset_axis)

def array_kw(*args, **kwargs):
    """ Define a Dimarray using keyword arguments for axes

    Parameters
    ----------
    *args : [values, [dims,]]
    **kwargs        : axes as keyword arguments

    Returns
    -------
    DimArray

    Notes
    -----
    The key-word functionality comes at the expense of metadata, which needs to be 
    added after creation of the DimArray object.

    If axes are passed as kwargs, `dims=` also needs to be provided
    or an error will be raised, unless values's shape is 
    sufficient to determine ordering (when all axes have different 
    sizes).  This is a consequence of the fact 
    that keyword arguments are *not* ordered in python (any order
    is lost since kwargs is a dict object)

    Axes passed by keyword arguments cannot have name already taken by other 
    parameters such as "values", "axes", "dims", "dtype" or "copy"

    Examples 
    --------

    >>> import dimarray as da
    >>> a = da.array_kw([[1,2,3],[4,5,6]], items=list("ab"), time=np.arange(1950,1953)) # here dims can be omitted because shape = (2, 3)
    >>> b = da.array_kw([[1,2,3],[4,5,6]], ['items','time'], items=list("ab"), time=np.arange(1950,1953)) # here dims can be omitted because shape = (2, 3)

    This is pretty similar to this other form:

    >>> c = da.DimArray([[1,2,3],[4,5,6]], {'items':list("ab"), 'time':np.arange(1950,1953)}) # here dims can be omitted because shape = (2, 3)
    >>> np.all(a == b) and np.all(a == c)
    True

    But note this would fail if both axes had the same size (would then need to specify the `dims` parameter).

    See also DimArray's doc for more examples

    See also:
    ---------
    DimArray.from_kw
    """
    return DimArray.from_kw(*args, **kwargs)

#array_kw.__doc__ = DimArray.from_kw.__doc__


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
    dimensions: 'unnamed', 'x0'
    0 / unnamed (2): 0 to 1
    1 / x0 (3): 0 to 2
    array([[1, 2, 3],
           [2, 4, 6]])

    From a dict, here also with axis alignment, and naming the axis

    >>> d = {'a':DimArray([10,20,30.],[0,1,2]), 'b':DimArray([1,2,3.],[1.,2.,3])}
    >>> a = da.array(d, keys=['a','b'], axis='items') # keys= just needed to enforce ordering
    >>> a
    dimarray: 6 non-null elements (2 null)
    dimensions: 'items', 'x0'
    0 / items (2): a to b
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
    dimensions: 'items', 'x0', 'x1'
    0 / items (2): a to b
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
	    data = align_axes(*data)

	if broadcast:
	    data = broadcast_arrays(*data) # make sure the arrays have the same dimension

	kwargs['align'] = False # already aligned

        return stack(data, *args, **kwargs) 

    # equivalent to calling DimArray
    else:
        return DimArray(data, *args, **kwargs)

# DEPRECATED
def from_arrays(*args, **kwargs):
    warnings.warn(FutureWarning('from_arrays is deprecated, use da.array() instead'))
    return array(*args, **kwargs)

def join(*args, **kwargs):
    warnings.warn(FutureWarning('join is deprecated, use da.array() instead'))
    return array(*args, **kwargs)

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
    dimensions: 'time', 'items'
    0 / time (2): 2000 to 2001
    1 / items (3): a to c
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

    return DimArray(values, axes=axes, dims=dims)

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
    dimensions: 'time', 'items'
    0 / time (2): 2000 to 2001
    1 / items (3): a to c
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]])
    """
    if dtype is None: dtype = a.dtype
    return empty(a.axes, dtype=dtype)

def nans(axes=None, dims=None, shape=None):
    """ Initialize an empty array filled with NaNs. See empty for doc.

    >>> nans(dims=('time','items'), shape=(2, 3))
    dimarray: 0 non-null elements (6 null)
    dimensions: 'time', 'items'
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
    dimensions: 'time', 'items'
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
    dimensions: 'time', 'items'
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

def hasnan(a):
    """ fast way of checking wether an array has nans

    Parameters
    ----------
     a: numpy array

    WARNING: if a is a not a numpy array, min will depend on the default behaviour 
    if the min method of these object, which may be to ignore nans (e.g. larry)
    """
    return np.isnan(a.min())
