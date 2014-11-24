""" collection of base obeje
"""
from __future__ import absolute_import
from collections import OrderedDict as odict
import warnings, copy
import numpy as np

import dimarray as da  # for the doctest, so that they are testable via py.test
from dimarray.decorators import format_doc
from dimarray.config import get_option

from .core import DimArray, array, Axis, Axes
from .core import align_axes, stack, concatenate
from .core.align import _check_stack_args, _get_axes, stack, concatenate, _check_stack_axis, get_dims as _get_dims, reindex_like
from .core.indexing import locate_many
from .core import pandas_obj
from .core.bases import AbstractDataset, GetSetDelAttrMixin, OpMixin

class Dataset(AbstractDataset, odict, OpMixin, GetSetDelAttrMixin):
# class Dataset(AbstractDataset, odict):
    """ Container for a set of aligned objects
    """
    _constructor = DimArray

    def __init__(self, *args, **kwargs):
        """ initialize a dataset from a set of objects of varying dimensions

        data  : dict of DimArrays or list of named DimArrays or Axes object
        keys  : keys to order data if provided as dict, or to name data if list
        """
        assert not {'axes','keys'}.issubset(kwargs.keys()) # just to check bugs due to back-compat ==> TO BE REMOVED AFTER DEBUGGING

        # check input arguments: same init as odict
        data = odict(*args, **kwargs)

        # Basic initialization
        self._axes = Axes()
        self._attrs = odict()

        # initialize an ordered dictionary
        super(Dataset, self).__init__()
        #self.data = odict()

        values = data.values()
        keys = data.keys()

        # Check everything is a DimArray
        #for key, value in zip(keys, values):
        for i, key in enumerate(keys):
            value = values[i]
            if not isinstance(value, DimArray):
                if np.isscalar(value):
                    values[i] = self._constructor(value)
                else:
                    raise TypeError("A Dataset can only store DimArray instances, got {}: {}".format(key, value))

        # Align objects
        values = align_axes(*values)

        # Append object (will automatically update self.axes)
        for key, value in zip(keys, values):
            self[key] = value

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, newaxes):
        for ax in newaxes:
            if ax.name in self.dims:
                self.axes[ax.name] = ax
            else:
                self.axes.append(ax)

    @property
    def dims(self):
        """ tuple of dimensions contained in the Dataset, consistently with DimArray's `dims`
        """
        return tuple([ax.name for ax in self._axes])

    @dims.setter
    def dims(self, newdims):
        """ rename all axis names at once
        """
        if not np.iterable(newdims): 
            raise TypeError("new dims must be iterable")
        if not len(newdims) == len(self.axes):
            raise ValueError("dimension mistmatch")

        # update every element's dimension
        for i, newname in enumerate(newdims):
            oldname = self.axes[i].name
            self.axes[i].name = newname

    @property
    def labels(self):
        """ tuple of axis values contained in the Dataset, consistently with DimArray's `labels`
        """
        return tuple([ax.values for ax in self.axes])

    #
    # overload dictionary methods
    #
    def __delitem__(self, item):
        """ 
        """
        axes = self[item].axes
        #del self.data[item]
        super(Dataset, self).__delitem__(item)
        #del super(Dataset, self)[item]

        # update axes
        for ax in axes:
            found = False
            for k in self:
                if ax.name in self[k].dims:
                    found = True
                    continue
            if not found:
                self.axes.remove(ax)

    def __setitem__(self, key, val):
        """ Make sure the object is a DimArray with appropriate axes

        Examples
        --------
        >>> ds = Dataset()
        >>> ds
        Dataset of 0 variables
        >>> a = DimArray([0, 1, 2], dims=('time',))
        >>> ds['yo'] = a 
        >>> ds['yo']
        dimarray: 3 non-null elements (0 null)
        0 / time (3): 0 to 2
        array([0, 1, 2])
        >>> ds['ya'] = a.values  # also accepts numpy array if shape matches
        >>> ds['ya']
        dimarray: 3 non-null elements (0 null)
        0 / time (3): 0 to 2
        array([0, 1, 2])
        """
        if not isinstance(val, DimArray):
            if np.isscalar(val):
                val = self._constructor(val)
            elif hasattr(val, '__array__'):
                if np.shape(val) == tuple([ax.size for ax in self.axes]):
                    val = self._constructor(val, axes=self.axes) # make a dimarray with same axes
                else:
                    raise ValueError("array_like shape does not match, use DimArray if dimensions vary within the dataset")
            else:
                raise TypeError("can only append DimArray instances")

        # shallow copy of the DimArray so that its axes attribute can be 
        # modified without affecting the original array
        val = copy.copy(val)  
        val._axes = copy.deepcopy(val.axes)

        # Check dimensions
        # make sure axes match those of the dataset
        for i, newaxis in enumerate(val.axes):

            # Check dimensions if already existing axis
            if newaxis.name in [ax.name for ax in self.axes]:
                existing_axis = self.axes[newaxis.name]
                if not newaxis == existing_axis:
                    raise ValueError("axes values do not match, align data first.\
                            \nDataset: {}, \nGot: {}".format(existing_axis, newaxis))

                # assign the Dataset axis : they all must share the same axis
                val.axes[i] = existing_axis

            # Append new axis
            else:
                self.axes.append(newaxis)  

            assert val.axes[i] is self.axes[newaxis.name]

        super(Dataset, self).__setitem__(key, val)

    def copy(self):
        ds2 = super(Dataset, self).copy() # odict method, copy axes but not metadata
        ds2.attrs.update(self.attrs)
        return ds2

    def __eq__(self, other):
        """ test equality but bypass annoying numpy's __eq__ method
        """
        return isinstance(other, Dataset) and self.keys() == other.keys() \
                and self.axes == other.axes \
                and np.all([np.all(self[k] == other[k]) for k in self.keys()])

    #
    # Backends
    #
    def write_nc(self, f, mode='w', clobber=True, format=None, **kwargs):
        """ Write Dataset to netCDF file.

        Wrapper around DatasetOnDisk

        Parameters
        ----------
        f : file name
        mode, clobber, format : seel netCDF4-python doc
        **kwargs : passed to netCDF4.Dataset.createVAriable (compression)
        """
        from dimarray.io.nc import DatasetOnDisk, nc, _maybe_open_file
        f, close = _maybe_open_file(f, mode=mode, clobber=clobber,format=format)
        store = DatasetOnDisk(f)
        # store = DatasetOnDisk(f, mode=mode, clobber=clobber, format=format)
        for name in self.keys():
            store.write(name, self[name], **kwargs)
        store.attrs.update(self.attrs) # attributes
        if isinstance(f, nc.Dataset): store.close() # do not close (deprecated)

    def write(self, *args, **kwargs):
        warnings.warn("Deprecated. Use write_nc.", FutureWarning)
        self.write_nc(*args, **kwargs)

    @classmethod
    def read_nc(cls, f, *args, **kwargs):
        """ Read dataset from netCDF file.
        """
        warnings.warn("Deprecated. Use dimarray.read_nc or dimarray.open_nc", FutureWarning)
        return da.io.nc.read_nc(f, *args, **kwargs)
    read = read_nc

    def to_array(self, axis=None, keys=None):
        """ Convert to DimArray

        axis  : axis name, by default "unnamed"
        """
        #if names is not None or dims is not None:
        #    return self.subset(names=names, dims=dims).to_array()

        if axis is None:
            axis = "unnamed"
            if axis in self.dims:
                i = 1
                while "unnamed_{}".format(i) in self.dims:
                    i+=1
                axis = "unnamed_{}".format(i)

        if axis in self.dims:
            raise ValueError("please provide an axis name which does not \
                    already exist in Dataset")

        if keys is None:
            keys = self.keys()

        # align all variables to the same dimensions
        data = odict()

        for k in keys:
            data[k] = self[k].reshape(self.dims).broadcast(self.axes)

        # make it a numpy array
        data = [data[k].values for k in keys]
        data = np.array(data)

        # determine axes
        axes = [Axis(keys, axis)] + self.axes 

        return self._constructor(data, axes)

    def take(self, names=None, indices=None, axis=0, indexing=None, tol=None, keepdims=False):
        """ Analogous to DimArray's take, but for each DimArray of the Dataset

        Parameters
        ----------
        names : list of variables to read, optional
        indices : int or list or slice (single-dimensional indices)
                   or a tuple of those (multi-dimensional)
                   or `dict` of { axis name : axis indices }
            Indices refer to Dataset axes. Any item that does not possess
            one of the dimensions will not be indexed along that dimension.
            For example, scalar items will be left unchanged whatever indices
            are provided.
        axis : None or int or str, optional
            if specified and indices is a slice, scalar or an array, assumes 
            indexing is along this axis.
        indexing : {'label', 'position'}, optional
            Indexing mode. 
            - "label": indexing on axis labels (default)
            - "position": use numpy-like position index
            Default value can be changed in dimarray.rcParams['indexing.by']
        tol : float, optional
            tolerance when looking for numerical values, e.g. to use nearest 
            neighbor search, default `None`.
        keepdims : bool, optional 
            keep singleton dimensions (default False)

        Returns
        -------
        Dataset

        See Also
        --------
        DimArrayOnDisk.read, DimArray.take

        Examples
        --------
        >>> a = DimArray([1,2,3], axes=('time', [1950, 1951, 1952]))
        >>> b = DimArray([11,22,33], axes=('time', [1951, 1952, 1953]))
        >>> ds = Dataset(a=a, b=b)
        >>> ds
        Dataset of 2 variables
        0 / time (4): 1950 to 1953
        a: ('time',)
        b: ('time',)
        >>> ds.take(indices=1951, axis='time')
        Dataset of 2 variables
        a: 2.0
        b: 11.0
        >>> ds.take(indices=0, axis='time', indexing='position')
        Dataset of 2 variables
        a: 1.0
        b: nan
        >>> ds['c'] = DimArray([[1,2],[11,22],[111,222],[3,4]], axes=[('time', [1950,1951,1952,1953]),('item',['a','b'])])
        >>> ds.take(indices={'time':1950})
        Dataset of 3 variables
        0 / item (2): 'a' to 'b'
        a: 1.0
        b: nan
        c: ('item',)
        >>> ds.take(indices={'time':1950})['c']
        dimarray: 2 non-null elements (0 null)
        0 / item (2): 'a' to 'b'
        array([1, 2])
        >>> ds.take(indices={'item':'b'})
        Dataset of 3 variables
        0 / time (4): 1950 to 1953
        a: ('time',)
        b: ('time',)
        c: ('time',)
        """
        # automatically read all variables to load (except for the dimensions)
        if names is None:
            names = self.keys()
        elif isinstance(names, basestring):
            raise TypeError("Please provide a sequence of variables to read.")

        tuple_indices = self._get_indices(indices, axis=axis, tol=tol, keepdims=keepdims, indexing=indexing)
        dict_indices = {dim:tuple_indices[i] for i, dim in enumerate(self.dims)}

        data = Dataset()
        # start with the axes, to make sure the ordering is maintained
        data.axes = self._getaxes_ortho(tuple_indices) 
        for nm in names:
            data[nm] = self[nm].take(indices={dim:dict_indices[dim] for dim in self[nm].dims}, indexing='position')
        data.attrs.update(self.attrs) # dataset's metadata
        return data

    def _apply_dimarray_axis(self, funcname, *args, **kwargs):
        """ Apply a function on every Dataset variable. 
        
        If the 'axis=' parameter is passed, only the variables with the required axis are called.
        """
        axis = kwargs.pop('axis',None)
        if axis is not None: axis = self.axes[axis].name
        kwargs['axis'] = axis

        d = self.to_odict()
        for k in self.keys():
            if axis is not None and axis not in self[k].dims: 
                continue
            #d[k] = self[k].apply(func, *args, **kwargs)
            d[k] = getattr(self[k], funcname)(*args, **kwargs)

        return Dataset(d)

    def mean(self, axis=0, **kwargs):
        """ Apply transformantion on every variable of the Dataset

        Examples
        --------
        >>> a = DimArray([1,2,3], axes=('time', [1950, 1951, 1952]))
        >>> b = DimArray([[11,22,33],[44,55,66]], axes=[('items',['a','b']), ('time', [1950, 1951, 1952])])
        >>> ds = Dataset(a=a, b=b)
        >>> ds.mean(axis='time')
        Dataset of 2 variables
        0 / items (2): 'a' to 'b'
        a: 2.0
        b: ('items',)
        >>> ds.mean(axis='items')
        Dataset of 2 variables
        0 / time (3): 1950 to 1952
        a: ('time',)
        b: ('time',)
        """
        return self._apply_dimarray_axis('mean', axis=axis, **kwargs)

    def std(self, axis=0, **kwargs): return self._apply_dimarray_axis('std', axis=axis, **kwargs)
    def var(self, axis=0, **kwargs): return self._apply_dimarray_axis('var', axis=axis, **kwargs)
    def median(self, axis=0, **kwargs): return self._apply_dimarray_axis('median', axis=axis, **kwargs)
    def sum(self, axis=0, **kwargs): return self._apply_dimarray_axis('sum', axis=axis, **kwargs)

    def to_dict(self):
        """ export to dict
        """
        return dict({nm:self[nm] for nm in self.keys()})

    def to_odict(self):
        """ export to ordered dict
        """
        return odict([(nm, self[nm]) for nm in self.keys()])

    def set_axis(self, values=None, axis=0, name=None, inplace=False, **kwargs):
        """ Set axis values, name and attributes of the Dataset
        
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
            (default False)
        **kwargs : key-word arguments
            Also reset other axis attributes, which can be single metadata
            or other axis attributes, via using `setattr`
            This includes special attributes `weights` and `attrs` (the latter
            reset all attributes)

        Returns
        -------
        Dataset instance, or None if inplace is True

        Notes
        -----
        This affects all DimArray present in the Dataset, since they share the same
        axes.

        Examples
        --------
        >>> ds = Dataset()
        >>> ds['a'] = da.zeros(shape=(3,))  # some dimarray with dimension 'x0'
        >>> ds['b'] = da.zeros(shape=(3,4)) # dimensions 'x0', 'x1'
        >>> ds.set_axis(['a','b','c'], axis='x0')
        Dataset of 2 variables
        0 / x0 (3): 'a' to 'c'
        1 / x1 (4): 0 to 3
        a: ('x0',)
        b: ('x0', 'x1')
        """
        if not inplace: self = self.copy()
        self.axes[axis].set(values=values, inplace=True, name=name, **kwargs)
        if not inplace: return self

    def reset(self, values=None, axis=0, name=None, **kwargs):
        "deprecated, see Dataset.set" 
        warnings.warn("Deprecated. Use Dataset.set", FutureWarning)
        if values is None: values = np.arange(self.size)
        if values is False: values = None
        return self.set(values, axis=axis, name=name, **kwargs)

    def rename_keys(self, mapper, inplace=False):
        """ Rename all variables in the Dataset

        Possible speedup compared to a classical dict-like operation 
        since an additional check on the axes is avoided.

        Parameters
        ----------
        mapper : dict-like or function to map oldname -> newname
        inplace : bool, optional
            if True, in-place modification, otherwise a copy with modified
            keys is retuend (default: False)

        Returns
        -------
        Dataset if inplace is False, otherwise None

        Examples
        --------
        >>> ds = da.Dataset(a=da.zeros(shape=(3,)), b=da.zeros(shape=(3,2)))
        >>> ds
        Dataset of 2 variables
        0 / x0 (3): 0 to 2
        1 / x1 (2): 0 to 1
        a: ('x0',)
        b: ('x0', 'x1')
        >>> ds.rename_keys({'b':'c'})
        Dataset of 2 variables
        0 / x0 (3): 0 to 2
        1 / x1 (2): 0 to 1
        a: ('x0',)
        c: ('x0', 'x1')
        """
        if inplace:
            ds = self
        else:
            ds = self.copy()

        if isinstance(mapper, dict):
            iterkeys = mapper.iteritems()
        else:
            if not callable(mapper):
                raise TypeError("mapper must be callable")
            iterkeys = [(old, mapper(old)) for old in ds.keys()]

        for old, new in iterkeys:
            val = super(Dataset, ds).__getitem__(old) # same as ds[old]
            super(Dataset, ds).__setitem__(new, val)
            super(Dataset, ds).__delitem__(old)

        if not inplace:
            return ds

    def rename_axes(self, mapper, inplace=False):
        """ Rename axes, analogous to rename_keys for axis names
        """
        if inplace:
            ds = self
        else:
            ds = self.copy()

        if isinstance(mapper, dict):
            iterkeys = mapper.iteritems()
        else:
            if not callable(mapper):
                raise TypeError("mapper must be callable")
            iterkeys = [(old, mapper(old)) for old in ds.dims]

        for old, new in iterkeys:
            ds.axes[old].name = new

        if not inplace:
            return ds

    def reindex_axis(self, values, axis=0, fill_value=np.nan, raise_error=False, method=None):
        """ analogous to DimArray.reindex_axis, but for a whole Dataset 

        See DimArray.reindex_axis for documention.
        """
        if isinstance(values, Axis):
            newaxis = values
            values = newaxis.values
            axis = newaxis.name
        elif np.isscalar(values) or type(values) is slice:
            raise TypeError("Please provide list, array-like or Axis object to perform re-indexing")
        else:
            values = np.asarray(values)

        # Get indices
        ax = self.axes[axis]
        # indices = ax.loc(values, mode='clip', side=method)
        indices = locate_many(ax.values, values, side=method or 'left')

        # Replace mismatch with missing values?
        mask = ax.values.take(indices) != values
        any_nan = np.any(mask)

        new = self.__class__()
        for k in self.keys():
            newobj = self[k].take_axis(indices, axis, indexing='position')
            if any_nan:
                if raise_error:
                    raise IndexError("Some values where not found in the axis: {}".format(values[mask]))
                if method is None:
                    newobj.put(mask, fill_value, axis=axis, inplace=True, indexing="position", cast=True)

                # Make sure the axis values match the requested new axis
                newobj.axes[axis][mask] = values[mask]
            new[k] = newobj
        new.attrs.update(self.attrs) # update metadata
        return new

    def reindex_like(self, other, **kwargs):
        """Analogous to DimArray.reindex_like

        >>> ds1 = da.Dataset(a=da.DimArray(axes=[[1,2,3]]))
        >>> ds2 = da.Dataset(b=da.DimArray(axes=[[1.,3.],['a','b']]))
        >>> ds2.reindex_like(ds1)
        Dataset of 1 variable
        0 / x0 (3): 1.0 to 3.0
        1 / x1 (2): 'a' to 'b'
        b: ('x0', 'x1')
        """
        return reindex_like(self, other, **kwargs)

    #
    # Operations
    #
    def _binary_op(self, func, other):
        """ generalize DimArray operation to a Dataset, for each key

        In case the keys differ, returns the intersection of the two datasets

        Just for testing:
        >>> ds = Dataset(b=DimArray([[0.,1],[1,2]]))
        >>> -ds
        Dataset of 1 variable
        0 / x0 (2): 0 to 1
        1 / x1 (2): 0 to 1
        b: ('x0', 'x1')
        >>> -ds["b"]
        dimarray: 4 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        1 / x1 (2): 0 to 1
        array([[-0., -1.],
               [-1., -2.]])
        >>> np.all(ds == ds)
        True
        >>> assert isinstance(-ds, Dataset)
        >>> assert isinstance(ds/0.5, Dataset)
        >>> assert isinstance(ds*0, Dataset)
        >>> (-ds -ds + ds/0.5 + ds*0+1)['b']
        dimarray: 4 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        1 / x1 (2): 0 to 1
        array([[ 1.,  1.],
               [ 1.,  1.]])
        >>> ds += 1
        >>> ds['b']
        dimarray: 4 non-null elements (0 null)
        0 / x0 (2): 0 to 1
        1 / x1 (2): 0 to 1
        array([[ 1.,  2.],
               [ 2.,  3.]])
        """
        assert isinstance(other, Dataset) or np.isscalar(other), "can only combine Datasets objects (func={})".format(func.__name__)
        # align all axes first
        reindex = get_option("op.reindex")
        if reindex and hasattr(other, 'axes') and other.axes != self.axes:
            other.reindex_like(self)
        # now proceed to operation
        res = self.__class__()
        for k1 in self.keys():
            if hasattr(other, 'keys'):
                for k2 in other.keys():
                    if k1 == k2:
                        res[k1] = self[k1]._binary_op(func, other[k2])
            else:
                res[k1] = self[k1]._binary_op(func, other)
        return res

    def _unary_op(self, func):
        res = self.__class__()
        for k in self.keys():
            res[k] = self[k]._unary_op(func)
        return res

def stack_ds(datasets, axis, keys=None, align=False):
    """ stack dataset along a new dimension

    Parameters
    ----------
    datasets: sequence or dict of datasets
    axis: str, new dimension along which to stack the dataset 
    keys, optional: stack axis values, useful if dataset is a sequence, or a non-ordered dictionary
    align, optional: if True, align axes (via reindexing) prior to stacking

    Returns
    -------
    stacked dataset

    See Also
    --------
    concatenate_ds

    Examples
    --------
    >>> a = DimArray([1,2,3], dims=('dima',))
    >>> b = DimArray([11,22], dims=('dimb',))
    >>> ds = Dataset({'a':a,'b':b}) # dataset of 2 variables from an experiment
    >>> ds2 = Dataset({'a':a*2,'b':b*2}) # dataset of 2 variables from a second experiment
    >>> stack_ds([ds, ds2], axis='stackdim', keys=['exp1','exp2'])
    Dataset of 2 variables
    0 / stackdim (2): 'exp1' to 'exp2'
    1 / dima (3): 0 to 2
    2 / dimb (2): 0 to 1
    a: ('stackdim', 'dima')
    b: ('stackdim', 'dimb')
    """
    #if not isinstance(axis, str): raise TypeError("axis parameter must be str")

    # make a sequence of datasets
    datasets, keys = _check_stack_args(datasets, keys) 

    # make sure the stacking dimension is ok
    dims = _get_dims(*datasets)
    axis = _check_stack_axis(axis, dims) 

    # find the list of variables common to all datasets
    variables = None
    for ds in datasets:
        # check that stack axis is not already present
        assert axis not in ds.dims, axis+" already exists in the dataset" 

        # check that variables have the same variables
        if variables is None:
            variables = ds.keys()
        else:
            assert sorted(ds.keys()) == sorted(variables), "variables differ across datasets"

    # Compute stacked dataset
    dataset = Dataset()
    for v in variables:
        arrays = [ds[v] for ds in datasets]
        array = stack(arrays, axis=axis, keys=keys, align=align)
        dataset[v] = array

    return dataset


def concatenate_ds(datasets, axis=0):
    """ concatenate two datasets along an existing dimension

    Parameters
    ----------
    datasets: sequence of datasets 
    axis: axis along which to concatenate

    Returns
    -------
    joint Dataset along axis

    NOTE: will raise an error if variables are there which do not contain the required dimension

    See Also
    --------
    stack_ds

    Examples
    --------
    >>> a = da.zeros(axes=[list('abc')], dims=('x0',))  # 1-D DimArray
    >>> b = da.zeros(axes=[list('abc'), [1,2]], dims=('x0','x1')) # 2-D DimArray
    >>> ds = Dataset({'a':a,'b':b}) # dataset of 2 variables from an experiment
    >>> a2 = da.ones(axes=[list('def')], dims=('x0',)) 
    >>> b2 = da.ones(axes=[list('def'), [1,2]], dims=('x0','x1')) # 2-D DimArray
    >>> ds2 = Dataset({'a':a2,'b':b2}) # dataset of 2 variables from a second experiment
    >>> concatenate_ds([ds, ds2])
    Dataset of 2 variables
    0 / x0 (6): 'a' to 'f'
    1 / x1 (2): 1 to 2
    a: ('x0',)
    b: ('x0', 'x1')
    """
    # find the list of variables common to all datasets
    variables = None
    for ds in datasets:

        # check that variables have the same variables
        if variables is None:
            variables = ds.keys()
        else:
            assert sorted(ds.keys()) == sorted(variables), "variables differ across datasets"

    # Compute concatenated dataset
    dataset = Dataset()
    for v in variables:
        arrays = [ds[v] for ds in datasets]
        array = concatenate(arrays, axis=axis)
        dataset[v] = array

    return dataset

