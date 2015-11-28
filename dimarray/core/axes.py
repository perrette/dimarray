from __future__ import absolute_import, division
import warnings
from collections import OrderedDict as odict
import string
import copy
import numpy as np

from dimarray.tools import is_DimArray, is_array1d_equiv, format_doc
from dimarray.core.bases import AbstractAxis, AbstractAxes, GetSetDelAttrMixin
from dimarray.core.indexing import _maybe_cast_type, is_monotonic

__all__ = ["Axis","Axes"]

# def _merge_sorted_arrays(a, b, drop_duplicates=True):
#     """Merge two sorted arrays
#
#     Parameters
#     ----------
#     a : array-like
#     b : array-like
#     drop_duplicates : bool, optional (default True)
#     """
#     i = np.searchsorted(a, b)
#     joined = np.insert(a, i, b)
#     if drop_duplicates:
#         joined = np.unique(joined)
#     return joined


def _check_axis_values(values, dtype=None):
    """ convert Axis type to have "object" instead of string
    """
    try:
        values = np.asarray(values, dtype=dtype)
    except Exception as error:
        raise TypeError(error.message + "\n==> axis values could not be converted to numpy array")

    # Treat the particular case of a sequence of sequences, leads to a 2-D array
    # ==> convert to a list of tuples
    if values.ndim == 2: 
        try:
            val = np.empty(values.shape[0], dtype=object)
            val[:] = zip(*values.T.tolist()) # pass a list of tuples
            values = val
        except:
            pass

    if values.ndim != 1:
        raise ValueError("an Axis object can only be 1-D, got ndim={}".format(values.ndim))

    if values.dtype.kind in ("S", "U"):
        values = np.asarray(values, dtype=object)

    return values

def _check_axes_merge(self, other):
    """Check self and other axes prior a merge operation:
    name and type must match
    """
    # name
    if isinstance(other, Axis):
        assert self.name == other.name, "axes have different names, cannot merge"
    else:
        other = Axis(other, self.name) # to give it the same methods is_monotonic etc...
    # type
    kind, consistent_kinds = _get_cast_kind(self.values.dtype.kind, other.values.dtype.kind)
    if self.dtype.kind != kind:
        self = self.cast(kind)
    if other.dtype.kind != kind:
        other = other.cast(kind)
    return self, other, consistent_kinds

def _get_cast_kind(kind0, kind1):
    """determine the kind to cast into
    """
    if kind0 == kind1:
        return kind0, True
    elif 'O' in (kind0, kind1):
        return 'O', False
    elif 'f' in (kind0, kind1): # float and integer
        return 'f', True
    elif 'i' in (kind0, kind1): # inclues unsigned?
        return 'i', True
    else:
        # in doubt...
        return 'O', False

#
# Axis class
#
class Axis(GetSetDelAttrMixin, AbstractAxis):
    """ Axis

    Attributes
    ----------
    values : numpy array (or list) 
    name : name (attribute)

    weights : [None] associated list of weights 
    tol : [None], if not None, attempt a nearest neighbour search with specified tolerance
    """
    __metadata_exclude__ = ['values','name','weights']

    def __init__(self, values, name="", weights=None, dtype=None, tol=None, **kwargs):
        self.name = name or getattr(values, "name", "")
        self._values = _check_axis_values(values, dtype)
        self.name = name 
        self.weights = weights # additional checks
        self._tol = tol
        self._attrs = odict()
        self._attrs.update(kwargs)
        self._monotonic = None

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        " init or update axis values, with size check in the second case"
        values = _check_axis_values(values)
        if self._values.size != values.size:
            raise ValueError("Invalid size. Expected: {}. Got: {}".format(self._values.size, values.size))
        self._values = values
        self._monotonic = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, basestring):
            raise TypeError("Axis name must be a string")
        if not name:
            raise ValueError("Axis name cannot be empty")
        self._name = name

    @property
    def tol(self):
        return self._tol
    @tol.setter
    def tol(self, val):
        self._tol = val

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, _weights):
        if _weights is None or callable(_weights):
            pass
        else:
            try:
                _weights = np.asarray(_weights)
            except:
                raise TypeError("weight must be array-like or callable, got: {}".format(_weights))
            if _weights.size != self.size:
                raise ValueError("weights must have the same size as axis values, got: {} and {} !".format(_weights.size, self.size))
        self._weights = _weights

    @weights.deleter
    def weights(self):
        self._weights = None

    def sort(self, *args, **kwargs):
        # in-place sort
        self._values.sort(*args, **kwargs)
        self._monotonic = True

    def __getitem__(self, item):
        """ access values elements & return an axis object
        """
        if type(item) is slice and item == slice(None):
            return self
        if not isinstance(item, slice) and not np.isscalar(item):
            item = np.asarray(item) # needed for boolean axes, otherwise problem
        values = self.values[item]
        if not isinstance(values, np.ndarray):
            return values # if collapsed to scalar, just return it
        if isinstance(self.weights, np.ndarray):
            weights = self.weights[item]
        else:
            weights = self.weights
        newaxis = Axis(values, self.name, weights=weights, tol=self.tol, **self.attrs)
        # slices keep the ordering
        if self._monotonic and type(item) is slice:
            newaxis._monotonic = self._monotonic
        return newaxis

    def __setitem__(self, item, value):
        """ do some type checking/conversion before setting new axis values

        Examples
        --------
        >>> a = Axis([1, 2, 3], name='dummy')
        >>> a.values
        array([1, 2, 3])
        >>> a[0] = 1.2  # convert to float
        >>> a.values
        array([ 1.2,  2. ,  3. ])
        >>> a[0] = 'a'  # convert to object dtype
        >>> a.values
        array(['a', 2.0, 3.0], dtype=object)
        """
        self._values = _maybe_cast_type(self._values, value)

        # now can proceed to asignment
        self._values[item] = value

        # here could do some additional check about _monotonic and other axis attributes
        # for now just set to None
        self._monotonic = None

    def take(self, indices, mode='raise'):
        """ Similar to numpy.take

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
            See help on numpy.take for more info.

        Returns
        -------
        subaxis : Axis instance
        """
        values = self._values.take(indices, mode=mode)
        if isinstance(self.weights, np.ndarray):
            weights = self.weights.take(indices, mode=mode)
        else:
            weights = self.weights
        return Axis(values, self.name, weights=weights, tol=self.tol, **self.attrs)


    def set(self, values=None, name=None, inplace=True, **kwargs):
        """ Set axis values and / or attributes

        Parameters
        ----------
        values : numpy array-like or mapper (callable or dict), optional
            - array-like : new axis values, must have exactly the same 
            length as original axis
            - dict : establish a map between original and new axis values
            - callable : transform each axis value into a new one
            - if None, axis values are left unchanged
            Default to None.
        name : str, optional
            give axis a new name
        inplace : bool, optional
            modify axis in-place (True) or return copy (False)? 
            (default True)
        **kwargs : key-word arguments
            Also reset other axis attributes, which can be single metadata
            or other axis attributes, via using `setattr`
            This includes special attributes `weights` and `attrs` (the latter
            reset all attributes)

        Returns
        -------
        None, or Axis instance inplace is False
        """
        if inplace: 
            ax = self
        else: 
            ax = self.copy()

        # check in put axis values
        if isinstance(values, dict):
            dict_ = values.copy()
            values = [dict_.pop(x, x) for x in self.values]
        elif callable(values):
            values = [values(x) for x in self.values]

        if values is not None:
            ax[:] = values # array-like of size axis.size
        if name is not None:
            ax.name = name
        if 'attrs' in kwargs:
            self.attrs = kwargs.pop('attrs')
        for k in kwargs:
            setattr(ax, k, kwargs[k])

        if not inplace: 
            return ax

    def reset(self, values=None, name=None, **kwargs):
        "deprecated, see Axis.set" 
        warnings.warn("Deprecated. Use Axis.set", FutureWarning)
        if values is None: values = np.arange(self.size)
        if values is False: values = None
        return self.set(values, axis=axis, name=name, **kwargs)

    def cast(self, dtype):
        " copy axis and cast into a new type"
        ax = Axis(np.asarray(self.values, dtype=dtype), self.name)
        ax.attrs.update(self.attrs)
        return ax

    def union(self, other):
        """Join two Axis objects
        
        Notes
        -----
        This removes duplicates by default

        Examples
        --------
        >>> ax1 = Axis([0, 1, 2, 3, 4], name='myaxis')
        >>> ax2 = Axis([-3, 2, 3, 6], name='myaxis')
        >>> ax3 = ax1.union(ax2)
        >>> ax3.values
        array([-3,  0,  1,  2,  3,  4,  6])
        """
        #assert isinstance(other, Axis), "can only make the Union of two Axis objects"
        self, other, consistent_kinds = _check_axes_merge(self, other)
        
        if np.all(self.values == other.values):
            # TODO: check other attributes such as weights
            return self.copy()
        elif self.values.size == 0:
            return other
        elif other.values.size == 0:
            return self

        def _same_slope(a, b):
            " both decreasing or both increasing "
            return (a[-1]>=a[0])==(b[-1]>=b[0])

        if consistent_kinds and self.is_monotonic() and other.is_monotonic() and _same_slope(self.values, other.values):
            # join two sorted axes
            joined = np.union1d(self.values, other.values)
            if self.values[-1] <= self.values[0]: # decreasing !
                joined = joined[::-1]

        else:
            # no ordering, just concatenate and drop doublons
            not_in_self = np.in1d(other.values, self.values, invert=True)
            joined = np.concatenate((self.values, other.values[not_in_self]))

        ax = Axis(joined, self.name)
        ax.attrs.update(self.attrs)
        return ax

    def intersection(self, other):
        """Join two Axis objects by taking the intersection
        """
        self, other, consistent_kinds = _check_axes_merge(self, other)

        if np.all(self.values == other.values):
            # TODO: check other attributes such as weights
            return self.copy()
        elif self.values.size == 0:
            return Axis([], self.name)
        elif other.values.size == 0:
            return Axis([], self.name)

        # numpy.intersect1d would be nicer but it would also sort the arrays...
        # restrict other with values in self
        in_self = np.in1d(other.values, self.values)
        oth = other.values[in_self]
        # restrict self values in restricted other
        in_other = np.in1d(self.values, oth)
        newval = self.values[in_other]
        
        ax = Axis(newval, self.name)
        ax.attrs.update(self.attrs)
        return ax
        # return Axis(np.intersect1d(self.values, other.values), self.name)

    def is_monotonic(self):
        """ return True if Axis is monotonic
        """
        if self._monotonic is None:
            self._monotonic = is_monotonic(self.values)
        return self._monotonic

    def __eq__(self, other):
        #return hasattr(other, "name") and hasattr(other, "values") and np.all(other.values == self.values) and self.name == other.name
        return isinstance(other, Axis) and np.all(other.values == self.values) and self.name == other.name

    def _bounds(self):
        if self.values.size == 0:
            start, stop = None, None
        else:
            start, stop = self.values[0], self.values[-1]
        return start, stop

    def copy(self):
        return copy.deepcopy(self) # deep copy: everything in the definition is copied

    @property
    def size(self): 
        return self.values.size

    @property
    def dtype(self): 
        return self.values.dtype

    @property
    def __len__(self): 
        return self.values.__len__

    @property
    def __array__(self): 
        return self.values.__array__

    def _get_weights(self, weights=None):
        """ return axis weights as a DimArray

        Parameters
        ----------
        weights : array-like or callable, optional
            if provided, will be used instead of self.weights
            used in weighted transformations (see doc in there)
        """
        from .dimarraycls import DimArray

        if weights is None:
            weights = self.weights

        # no weights
        if weights is None:
            weights = np.ones_like(self.values)

        # function of axis
        elif callable(weights):
            weights = weights(self.values)

        # same weight for every element
        elif np.size(weights) == 1:
            weights = np.zeros_like(self.values) + weights

        # already an array of weights
        else:
            weights = np.asarray(weights)

        # index on one dimension
        ax = Axis(self.values, name=self.name)

        return DimArray(weights, [ax])

    def to_pandas(self):
        """ convert to pandas Index
        """
        import pandas as pd
        return pd.Index(self.values, name=self.name)

    @classmethod
    def from_pandas(cls, index):
        return cls(index.values, name=index.name)

    @classmethod
    def as_axis(cls, ax, defaultname=""):
        """ convert to Axis instance, more general than __init__
        """
        if isinstance(ax, Axis):
            pass
        elif isinstance(ax, tuple) or isinstance(ax, list) and len(ax) == 2:
            ax = cls(ax[1], ax[0]) # values, name
        else:
            try:
                ax = cls(np.asarray(ax), getattr(ax,'name',defaultname))
            except Exception as error: 
                print error.message
                raise TypeError("Cannot convert to Axis object: {}. \nPlease provide an Axis instance or (name, values) tuple".format(type(ax)))
        return ax


class MultiAxis(Axis):
    """ an Axis which is a grouping of several axes flattened together
    """
    __metadata_exclude__ = Axis.__metadata_exclude__ + ['axes']

    def __init__(self, *axes):
        """
        """
        self.axes = Axes(axes)
        self._name = ",".join([ax.name for ax in self.axes])
        self._values = None  # values not computed unless needed
        self._weights = None  
        self._size = None  
        self._attrs = odict()

    @property
    def values(self):
        """ values as 2-D numpy array, to keep things consistent with Axis
        """
        if self._values is None:
            self._values = self._get_values()
        return self._values

    def _get_values(self):
        # Each element of the new axis is a tuple, which makes a 2-D numpy array
        if len(self.axes) == 1:
            return self.axes[0].values

        aval = _flatten(*[ax.values for ax in self.axes])
        val = np.empty(aval.shape[0], dtype=object)
        val[:] = zip(*aval.T.tolist()) # pass a list of tuples
        return val 

    @property
    def weights(self):
        """ Combine the weights as a product of the individual axis weights
        """
        if self._weights is None:
            #self._weights = self._get_weights()
            if np.all([ax.weights is None for ax in self.axes]):
                self._weights = None
            else:
                self._weights =_flatten(*[ax._get_weights() for ax in self.axes]).prod(axis=1)
        return self._weights

    @property
    def size(self): 
        """ size as product of axis sizes
        """
        if self._size is None:
            self._size = self._get_size()
        return self._size

    def _get_size(self):
        return np.prod([ax.size for ax in self.axes])

    @property
    def levels(self):
        """ for convenience, return individual axis values (like pandas)
        """
        return [ax.values for ax in self.axes]

    def __str__(self):
        return ",".join([str(ax) for ax in self.axes])

    def to_MultiIndex(self):
        """ convert to pandas MultiIndex
        """
        import pandas as pd
        index = pd.MultiIndex.from_tuples(self.values, names=[ax.name for ax in self.axes])
        index.name = self.name
        return index

    to_pandas = to_MultiIndex # over load Axis method

    @classmethod
    def from_MultiIndex(cls, mi):
        """ from a pandas MultiIndex
        """
        axes = []
        for i, lev in enumerate(mi.levels):
            nm = lev.name
            if nm is None: 
                nm = "lev{}".format(i)
            axes.append(Axis(lev.values, nm))
        ax = cls(*axes)
        if mi.name is not None:
            ax.name = mi.name
        return ax

    def sort(self, *args, **kwargs):
        raise NotImplementedError()

def _flatten(*list_of_arrays):
    """ flatten a list of arrays ax1, ax2, ... to  a list of tuples [(ax1[0], ax2[0], ax3[0]..), (ax1[0], ax2[0], ax3[1]..), ...]
    """
    assert len(list_of_arrays) > 0, "empty axis"
    if len(list_of_arrays) == 1:
        return list_of_arrays[0]
    kwargs = dict(indexing="ij")
    grd = np.meshgrid(*list_of_arrays, **kwargs)
    array_of_tuples = np.array(zip(*[g.ravel() for g in grd]))
    assert array_of_tuples.shape[1] == len(list_of_arrays), "pb when reshaping: {} and {}".format(array_of_tuples.shape, len(list_of_arrays))
    assert array_of_tuples.shape[0] == np.prod([x.size for x in list_of_arrays]), "pb when reshaping: {} and {}".format(array_of_tuples.shape, np.prod([x.size for x in list_of_arrays]))
    return array_of_tuples

#
# List of axes
#

class Axes(AbstractAxes, list):
    """ Axes class: inheritates from a list but dict-like access methods for convenience
    """
    _Axis = Axis # to use certain AbstractAxes methods
    def __init__(self, *list_):
        """Initialize Axes via a list of Axis-compatible objects
        """
        list.__init__(self)
        for v in list(*list_):
            self.append(v)

    @staticmethod
    def _init(*args, **kwargs):
        axes = _init_axes(*args, **kwargs)
        return axes

    @classmethod
    def from_shape(cls, shape, dims=None):
        """ return default axes based on shape
        """
        axes = cls()
        for i,ni in enumerate(shape):
            if dims is None:
                name = "x{}".format(i) # default name
            else:
                name = dims[i]
            axis = Axis(np.arange(ni), name)
            axes.append(axis)

        return axes

    @classmethod
    def from_arrays(cls, arrays, dims=None):
        """  list of np.ndarrays and dims
        """
        assert np.iterable(arrays) and (dims is None or len(dims) == len(arrays)), "invalid input arrays={}, dims={}".format(arrays, dims)

        # default names
        if dims is None: 
            dims = ["x{}".format(i) for i in range(len(arrays))]

        return cls(zip(dims, arrays))

    @classmethod
    def from_dict(cls, kwaxes, dims=None, shape=None, check_order=True):
        """ infer dimensions from key-word arguments
        """
        # if no key-word argument is given, just return default axis
        if len(kwaxes) == 0:
            return cls.from_shape(shape, dims)

        axes = cls()
        for k in kwaxes:
            axes.append(Axis(kwaxes[k], k))

        # Make sure the order is right (since it is lost via dict-passing)

        # preferred solution: dims is given
        if dims is not None:
            axes.sort(dims)

        # alternative option: only the shape is given
        elif shape is not None:
            assert len(shape) == len(kwaxes), "shape does not match kwaxes !"
            current_shape = [ax.size for ax in axes]
            assert set(shape) == set(current_shape), "mismatch between array shape and axes"
            assert len(set(shape)) == len(set(current_shape)) == len(set([ax.name for ax in axes])), \
    """ some axes have the same size !
    ==> ambiguous determination of dimensions order via keyword arguments only
    ==> explictly supply `dims=` or use  Axes() or from_arrays() methods" """
            argsort = [current_shape.index(k) for k in shape]

            assert len(argsort) == len(axes), "keyword arguments do not match shape !"
            axes = Axes([axes[i] for i in argsort])

            current_shape = tuple([ax.size for ax in axes])
            assert current_shape == shape, "dimensions mismatch (axes shape: {} != values shape: {}".format(current_shape, shape)

        elif check_order:
            #warnings.warn("no shape information: random axis order")
            raise ValueError("no shape information: random order")

        dims = [ax.name for ax in axes]
        assert len(set(dims)) == len(dims), "what's wrong??"

        return axes

    #
    # Overload basic list methods
    #
    def append(self, newax):
        " append new axis "
        newax = Axis.as_axis(newax, defaultname="x{}".format(len(self)))
        if newax.name in [ax.name for ax in self]:
            raise ValueError("axis name already exist: {}".format(newax.name))
        list.append(self, newax)

    def __getitem__(self, k):
        " get an axis by integer or name "
        if isinstance(k, basestring):
            dims = [ax.name for ax in self]
            try:
                k = dims.index(k)
            except IndexError:
                # common operation: make a clear error message
                IndexError("Axis name not found: {}. Existing dimensions : {}".format(k, dims))
        return list.__getitem__(self, k)

    def __setitem__(self, k, newax):
        """ update existing axis, the size cannot be changed
        """
        if isinstance(k, basestring):
            k = [ax.name for ax in self].index(k)
        curax = list.__getitem__(self, k)

        if not isinstance(newax, Axis):
            newax = Axis(newax, getattr(newax, 'name', curax.name))

        # NOTE: think about it. Is that check necessary?
        if newax.size != curax.size:
            raise ValueError("set axis: size mismatch.\nExpected: {}, got: {}".format(curax.size, newax.size))

        list.__setitem__(self, k, newax)

    def insert(self, pos, ax):
        if not isinstance(ax, Axis):
            raise TypeError("Must be Axis instance, got {}".format(type(ax)))
        list.insert(self, pos, ax)

    def pop(self, axis):
        pos = self._get_idx(axis)
        return list.pop(self, pos)

    def sort(self, dims):
        """ sort IN PLACE according to the order in "dims"
        """
        if type(dims[0]) is int:
            dims = [ax.name for ax in self]

        #list.sort(self, key=lambda x: dims.index(x.name))
        super(Axes, self).sort(key=lambda x: dims.index(x.name))

    # 
    # a few extras
    # 
    def copy(self):
        return copy.deepcopy(self) # not only the list but the elements of the list are copied

    def _get_idx(self, axis):
        " always return axis integer location "
        if isinstance(axis, basestring):
            dims = [ax.name for ax in self]
            try:
                axis = dims.index(axis)
            except IndexError:
                IndexError("Axis name not found: {}. Existing dimensions : {}".format(axis, dims))
        return axis

def _init_axes(axes=None, dims=None, labels=None, shape=None, check_order=True):
    """ initialize axis instance with many different ways

    axes:
        - dict
        - list of Axis objects
        - list of tuples `dim, array`
        - list of arrays, to be complemented by "dims="
        - nothing

    dims: tuple or list of dimension names
    shape
    """
    #back compat
    if axes is None:
        axes = labels 

    # special case: 1D object: accept single axis instead of list of axes/dimensions
    if shape is not None and len(shape) == 1:
        
        # accept a tuple ('dim', axis values) instead of [(...)]
        if type(axes) is tuple:
            if len(axes) == 2 and type(axes[0]) is str and is_array1d_equiv(axes[1]):
                axes = [axes]

        # accept axes=axis, dims='dim' (instead of list)
        elif (axes is None or is_array1d_equiv(axes)) and (type(dims) in (str, type(None))):
            #if axes is not None: assert np.size(axes) == values.size, "invalid argument: bad size"
            axes = [axes] if axes is not None else None
            dims = [dims] if dims is not None else None

    # axis not provided: check whether values has an axes field
    if axes is None:
        if shape is None:
            if dims is None:
                shape = ()
            else:
                shape = [0 for i in dims] # add 0-size axes

            #raise ValueError("at least shape must be provided (if axes are not)")

        # define a default set of axes if not provided
        axes = Axes.from_shape(shape, dims=dims)
        return axes

    elif isinstance(axes, dict):
        kwaxes = axes
        if isinstance(kwaxes, odict) and dims is None:
            dims = kwaxes.keys()
        axes = Axes.from_dict(kwaxes, dims=dims, shape=shape, check_order=check_order)
        return axes

    else:
        if not isinstance(axes, list) and not isinstance(axes, tuple):
            raise TypeError("axes, if provided, must be a list of: `Axis` or `tuple` or arrays. Got: {} (instance:{})".format(axes.__class__, axes))

    # FROM HERE list or tuple

    # empty axis
    if len(axes) == 0:
        return Axes()

    # list of Axis objects
    elif np.all([isinstance(ax, Axis) for ax in axes]):
        axes = Axes(axes)

    # (name, values) tuples
    elif np.all([isinstance(ax, tuple) for ax in axes]):
        axes = Axes(axes)

    # axes contains only axis values, with names possibly provided in `dims=`
    elif np.all([type(ax) in (list, np.ndarray) for ax in axes]):
        axes = Axes.from_arrays(axes, dims=dims)

    # axes only cointain axis labels
    elif np.all([type(ax) in (str, unicode) for ax in axes]):
        axes = Axes.from_shape(shape, dims=axes)

    else:
        raise TypeError("axes, if provided, must be a list of: `Axis` or `tuple` or arrays. Got: {} (instance:{})".format(axes.__class__, axes))

    return axes
