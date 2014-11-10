""" collection of base obeje
"""
from collections import OrderedDict as odict
import warnings, copy
import numpy as np

import dimarray as da  # for the doctest, so that they are testable via py.test
from dimarray.decorators import format_doc

from core import DimArray, array, Axis, Axes
from core import align_axes, stack, concatenate
from core.align import _check_stack_args, _get_axes, stack, concatenate, _check_stack_axis, get_dims as _get_dims
from core import pandas_obj
from core.metadata import MetadataBase
from core.axes import _doc_reset_axis
from core.prettyprinting import repr_dataset

class Dataset(odict, MetadataBase):
    """ Container for a set of aligned objects
    """
    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = axes

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
        self.axes = Axes()

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
    def dims(self):
        """ tuple of dimensions contained in the Dataset, consistently with DimArray's `dims`
        """
        return tuple([ax.name for ax in self.axes])

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

            # axes in individual items will be updated automatically 
            # since they are all references of the central axes

    @property
    def labels(self):
        """ tuple of axis values contained in the Dataset, consistently with DimArray's `labels`
        """
        return tuple([ax.values for ax in self.axes])

    __repr__ = repr_dataset

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
        <BLANKLINE>
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
        val.axes = copy.deepcopy(val.axes)

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

        # now just checking 
        test_internal = super(Dataset, self).__getitem__(key)
        for ax in test_internal.axes:
            assert self.axes[ax.name] is ax

    def copy(self):
        ds2 = super(Dataset, self).copy() # odict method, copy axes but not metadata
        ds2._metadata(self._metadata())
        return ds2

    def __eq__(self, other):
        """ test equality but bypass annoying numpy's __eq__ method
        """
        return isinstance(other, Dataset) and self.keys() == other.keys() \
                and self.axes == other.axes \
                and np.all([np.all(self[k] == other[k]) for k in self.keys()])

    #
    #
    #
    def write_nc(self, f, *args, **kwargs):
        """ Save dataset in netCDF file.

        If you see this documentation, it means netCDF4 is not installed on your system 
        and you will not be able to use this functionality.
        """
        import io.nc as ncio
        ncio._write_dataset(f, self, *args, **kwargs)

    write = write_nc

    @classmethod
    def read_nc(cls, f, *args, **kwargs):
        """ Read dataset from netCDF file.

        If you see this documentation, it means netCDF4 is not installed on your system 
        and you will not be able to use this functionality.
        """
        import io.nc as ncio
        return ncio._read_dataset(f, *args, **kwargs)

    #read = read_nc

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

    # 
    # REMOVE THESE FUNCTIONS AS NON-ESSENTIAL ???
    #
    def take(self, indices, axis=0, raise_error=False, **kwargs):
        """ analogous to DimArray's take, but for each DimArray of the Dataset

        Parameters
        ----------
        indices : scalar, or array-like, or slice
        axis : axis name (str)
        raise_error : raise an error if a variable does not have the desired dimension
        **kwargs : arguments passed to the axis locator, similar to `take`, such as `indexing` or `keepdims`

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
        >>> ds.take(1951, axis='time')
        Dataset of 2 variables
        <BLANKLINE>
        a: 2.0
        b: 11.0
        >>> ds.take(0, axis='time', indexing='position')
        Dataset of 2 variables
        <BLANKLINE>
        a: 1.0
        b: nan
        >>> ds['c'] = DimArray([[1,2],[11,22],[111,222],[3,4]], axes=[('time', [1950,1951,1952,1953]),('item',['a','b'])])
        >>> ds.take({'time':1950})
        Dataset of 3 variables
        0 / item (2): a to b
        a: 1.0
        b: nan
        c: ('item',)
        >>> ds.take({'time':1950})['c']
        dimarray: 2 non-null elements (0 null)
        0 / item (2): a to b
        array([1, 2])
        >>> ds.take({'item':'b'})
        Dataset of 3 variables
        0 / time (4): 1950 to 1953
        a: ('time',)
        b: ('time',)
        c: ('time',)
        """
        # first find the index for the shared axes
        kw_indices = {self.axes[i].name:ind for i,ind in enumerate(self.axes.loc(indices, axis=axis, **kwargs))}

        # then apply take in 'position' mode
        newdata = self.__class__()
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
        0 / items (2): a to b
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

    def __getattr__(self, att):
        """ allow access of dimensions
        """
        # check for dimensions
        if att in self.dims:
            ax = self.axes[att]
            return ax.values # return numpy array

        else:
            raise AttributeError("{} object has no attribute {}".format(self.__class__.__name__, att))

    def to_dict(self):
        """ export to dict
        """
        return dict({nm:self[nm] for nm in self.keys()})

    def to_odict(self):
        """ export to ordered dict
        """
        return odict([(nm, self[nm]) for nm in self.keys()])

    @format_doc(**_doc_reset_axis)
    def set_axis(self, values=None, axis=0, inplace=False, **kwargs):
        """ (re)set axis values and attributes in all dimarrays present in the dataset

        Parameters
        ----------
        {values}
        {axis}
        {inplace}
        {kwargs}

        Returns
        -------
        Dataset instance, or None if inplace is True

        Examples
        --------
        >>> ds = Dataset()
        >>> ds['a'] = da.zeros(shape=(3,))  # some dimarray with dimension 'x0'
        >>> ds['b'] = da.zeros(shape=(3,4)) # dimensions 'x0', 'x1'
        >>> ds.set_axis(['a','b','c'], axis='x0')
        Dataset of 2 variables
        0 / x0 (3): a to c
        1 / x1 (4): 0 to 3
        a: ('x0',)
        b: ('x0', 'x1')
        """
        if inplace is False:
            self = self.copy()
        ## update every dimarray in the dict
        #axis_name = self.axes[axis].name
        #for nm in self.keys():
        #    if not axis_name in self[nm].dims:
        #        continue
        #    super(Dataset, self).__setitem__(nm, self[nm].set_axis(values, axis, inplace=False, **kwargs) )

        # update the main axis instance
        self.axes = self.axes.set_axis(values, axis, inplace=False, **kwargs)

        if inplace is False:
            return self

    @format_doc(**_doc_reset_axis)
    def reset_axis(self, axis=0, inplace=False, **kwargs):
        """ (re)set axis values and attributes in all dimarrays present in the dataset

        Parameters
        ----------
        {axis}
        {inplace}
        {kwargs}

        Returns
        -------
        Dataset instance, or None if inplace is True

        Examples
        --------
        >>> ds = Dataset()
        >>> ds['a'] = da.zeros(axes=[['a','b','c']])  # some dimarray with dimension 'x0'
        >>> ds['b'] = da.zeros(axes=[['a','b','c'], [11,22,33,44]]) # dimensions 'x0', 'x1'
        >>> ds.reset_axis(axis='x0')
        Dataset of 2 variables
        0 / x0 (3): 0 to 2
        1 / x1 (4): 11 to 44
        a: ('x0',)
        b: ('x0', 'x1')
        """
        if inplace is False:
            self = self.copy()

        ## update every dimarray in the dict
        #axis_name = self.axes[axis].name
        #for nm in self.keys():
        #    if not axis_name in self[nm].dims:
        #        continue
        #    super(Dataset, self).__setitem__(nm, self[nm].reset_axis(axis, inplace=False, **kwargs) )

        # update the main axis instance
        self.axes = self.axes.reset_axis(axis, inplace=False, **kwargs)

        if inplace is False:
            return self


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
    0 / stackdim (2): exp1 to exp2
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
    0 / x0 (6): a to f
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

