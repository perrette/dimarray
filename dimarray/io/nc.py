""" NetCDF I/O to access and save geospatial data
"""
import os
import glob, copy
from collections import OrderedDict as odict
from functools import partial
import warnings
import numpy as np
import netCDF4 as nc
import dimarray as da
#from geo.data.index import to_slice, _slice3D

from dimarray.compat.pycompat import basestring, zip
from dimarray.decorators import format_doc
from dimarray.dataset import Dataset, concatenate_ds, stack_ds
from dimarray.core import DimArray, Axis, Axes
from dimarray.config import get_option
from dimarray.core.bases import AbstractDimArray, AbstractDataset, AbstractAxis, GetSetDelAttrMixin, AbstractAxes
# from dimarray.core.metadata import _repr_metadata
from dimarray.core.prettyprinting import repr_axis, repr_axes, repr_dimarray, repr_dataset, repr_attrs

__all__ = ['read_nc','summary_nc', 'write_nc']


#
# Global variables 
#
FORMAT = get_option('io.nc.format') # for the doc

#
# Helper functions
#
def _maybe_convert_dtype(values):
    """ strings are given "object" type in Axis object
    ==> assume all objects are actually strings
    NOTE: this will fail for other object-typed axes such as tuples
    """
    # if dtype is np.dtype('O'):
    values = np.asarray(values)
    dtype = values.dtype
    if dtype > np.dtype('S1'): # all strings, this include objects dtype('O')
        values = np.asarray(values, dtype=object)
        dtype = str
    return values, dtype

#
# Define an open_nc function return an on-disk Dataset object, more similar to 
# netCDF4's Dataset.
#

#
# Specific to NetCDF I/O
#
class NetCDFOnDisk(object):
    " DimArray wrapper for netCDF4 Variables and Datasets "
    @property
    def nc(self):
        return self._ds
    @property
    def _obj(self):
        return NotImplementedError("need to be subclassed")
    @property
    def attrs(self):
        return AttrsOnDisk(self._obj)
    def close(self):
        return self._ds.close()
    def __exit__(self, exception_type, exception_value, tracebook):
        self.close()
    def __enter__(self):
        return self


class DatasetOnDisk(GetSetDelAttrMixin, NetCDFOnDisk, AbstractDataset):
    """ Dataset on Disk

    .. versionadded :: 0.2

    See open_nc for examples of use.
    """
    @format_doc(format=FORMAT)
    def __init__(self, f, mode="r", clobber=True, diskless=False, persist=False, format=FORMAT, **kwargs):
        """ 
        Parameters
        ----------
        f : file name or netCDF4.Dataset instance
        mode : str, optional
            access mode. This include "r" (read) (the default), "w" (write)
            and "a" (append)
            See netCDF4.Dataset for full information. 
        clobber : `bool`, optional
            if True (default), opening a file with mode='w'
            will clobber an existing file with the same name.  if False, an
            exception will be raised if a file with the same name already exists.
        diskless : see netCDF4.Dataset help
        persist : see netCDF4.Dataset help
        format : `str`
            netCDF file format. Default is '{format}' (only accounted for during file creation)

        **kwargs : key-word arguments, used as default argument when creating a variable
    
        Notes
        -----
        See the netCDF4-python module documentation for more information about the use
        of keyword arguments to DatasetOnDisk
        """
        if isinstance(f, nc.Dataset):
            self._ds = f
        else:
            self._ds = nc.Dataset(f, mode=mode, clobber=clobber, diskless=diskless, persist=persist, format=format)
        # assert self._kwargs is not None
        self._kwargs = kwargs
    @property
    def _obj(self):
        return self._ds

    def read(self, names=None, indices=None, axis=0, indexing=None, tol=None, keepdims=False):
        """ Read values from disk

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
        open_nc : examples of use
        DimArrayOnDisk.read, DatasetOnDisk.write, DimArray.take
        """
        # automatically read all variables to load (except for the dimensions)
        if names is None:
            names = self.keys()
        elif isinstance(names, basestring):
            return self[names].read(indices=indices, axis=axis, indexing=indexing, tol=tol, keepdims=keepdims)
        # else:
        #     raise TypeError("Expected list or str for 'names=', got {}".format(names))

        tuple_indices = self._get_indices(indices, axis=axis, tol=tol, keepdims=keepdims, indexing=indexing)
        dict_indices = {dim:tuple_indices[i] for i, dim in enumerate(self.dims)}

        data = Dataset()
        # start with the axes, to make sure the ordering is maintained
        data.axes = self._getaxes_ortho(tuple_indices) 
        for nm in names:
            data[nm] = self[nm].read(indices={dim:dict_indices[dim] for dim in self[nm].dims}, indexing='position')
        data.attrs.update(self.attrs) # dataset's metadata
        return data

    def __getitem__(self, name):
        if name in self.dims:
            raise KeyError("Use 'axes' property to access dimension variables: {}".format(name))
        if name not in self.keys():
            raise KeyError("Variable not found in the dataset: {}.\nExisting variables: {}".format(name, self.keys()))
        return DimArrayOnDisk(self._ds, name)


    # def write(self, name, dataset, zlib=False, **kwargs):
    def write(self, name, dima, **kwargs):
        """ Write a variable to a netCDF4 dataset.

        Parameters
        ----------
        name : variable name 
        dima : DimArray instance
        zlib : `bool`
            Enable zlib compression if True. Default is False (no compression).
        complevel : `int`
            integer between 1 and 9 describing the level of compression desired. Ignored if zlib=False.
        **kwargs : key-word arguments
            Any additional keyword arguments accepted by `netCDF4.Dataset.createVariable`

        Notes
        -----
        See the netCDF4-python module documentation for more information about the use
        of keyword arguments to write_nc.

        See also
        --------
        DimArrayOnDisk.write

        Examples
        --------
        >>> import numpy as np
        >>> import dimarray as da

        Create a DimArray (in memory), with metadata

        >>> dima = da.DimArray([[1,2,3],[4,5,6]], axes=[('time',[2000,2045.5]),('scenario',['a','b','c'])])
        >>> dima.units = 'myunits' # metadata 
        >>> dima.axes['time'].units = 'metadata-dim-in-memory'

        Write it to disk, and add some additional metadata

        >>> ds = da.open_nc('/tmp/test.nc', mode='w')
        >>> ds['myvar'] = dima
        >>> ds['myvar'].bla = 'bla'
        >>> ds['myvar'].axes['time'].yo = 'metadata-dim-on-disk'
        >>> ds.axes['scenario'].ya = 'metadata-var-on-disk'
        >>> ds.yi = 'metadata-dataset-on-disk'
        >>> ds.close()

        Check the result with ncdump utility from a terminal (need to be installed for the test below to work)
        
        :> ncdump -h /tmp/test.nc
        netcdf test {
        dimensions:
            time = 2 ;
            scenario = 3 ;
        variables:
            double time(time) ;
                time:units = "metadata-dim-in-memory" ;
                time:yo = "metadata-dim-on-disk" ;
            string scenario(scenario) ;
                scenario:ya = "metadata-var-on-disk" ;
            int64 myvar(time, scenario) ;
                myvar:units = "myunits" ;
                myvar:bla = "bla" ;

        // global attributes:
                :yi = "metadata-dataset-on-disk" ;
        }
        """
        # if not isinstance(obj, da.DimArray):
        #     raise TypeError("Can only write Dataset, use `ds[name] = dima` to write a DimArray")
        _, nctype = _maybe_convert_dtype(dima)

        name = name or getattr(self, "name", None)
        if not name:
            raise ValueError("Need to provide variable name")

        if name not in self._ds.variables.keys():
            if np.isscalar(dima):
                dima = da.DimArray(dima)
            if not isinstance(dima, DimArray):
                raise TypeError("Expected DimArray, got {}".format(type(dima)))
            if dima.ndim > 0:
                # create Dimension and associated variable
                for ax in dima.axes:
                    if ax.name not in self.dims:
                        self.axes.append(ax, **kwargs)

            kw = self._kwargs.copy() 
            kw.update(kwargs)
            self._ds.createVariable(name, nctype, dima.dims, **kw)

        DimArrayOnDisk(self._ds, name)[()] = dima

    __setitem__ = write

    def __delitem__(self, name):
        " will raise an Exception if on-disk "
        del self._ds.variables[name]


    def keys(self):
        " all variables except for dimensions"
        return [nm for nm in self._ds.variables.keys() if nm not in self.dims]

    def values(self):
        return [self[k] for k in self.keys()]

    def __len__(self):
        return len(self.keys())

    @property
    def dims(self):
        return tuple(self._ds.dimensions.keys())
    @property
    def axes(self):
        return AxesOnDisk(self._ds, self.dims, **self._kwargs)
    @axes.setter
    def axes(self, newaxes):
        for ax in newaxes:
            if ax.name in self.dims:
                self.axes[ax.name] = ax
            else:
                self.axes.append(ax)

    _repr = repr_dataset

    def __iter__(self):
        for k in self.keys():
            yield k

class NetCDFVariable(NetCDFOnDisk):
    @property
    def _obj(self):
        return self.values
    @property
    def values(self):
        return self._ds.variables[self._name] # simply the variable to be indexed and returns values like netCDF4
    @values.setter
    def values(self, values):
        self[()] = values
    def __array__(self):
        return self.values[()] # returns a numpy array
    
class DimArrayOnDisk(GetSetDelAttrMixin, NetCDFVariable, AbstractDimArray):
    _constructor = DimArray
    _broadcast = False
    def __init__(self, ds, name, _indexing=None):
        self._indexing = _indexing or get_option('indexing.by')
        self._ds = ds
        self._name = name
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, newname):
        self._ds.renameVariable(self, self._name, newname)
        self._name = newname
    @property
    def dims(self):
        return tuple(self._obj.dimensions)
    @property
    def axes(self):
        return AxesOnDisk(self._ds, self.dims)

    def write(self, indices, values, axis=0, indexing=None, tol=None):
        """ Write numpy array or DimArray to netCDF file (The 
        variable must have been created previously)

        Parameters
        ----------
        indices : int or list or slice (single-dimensional indices)
                   or a tuple of those (multi-dimensional)
                   or `dict` of { axis name : axis values }
        values : np.ndarray or DimArray
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

        See Also
        --------
        DimArray.put, DatasetOnDisk.write, DimArrayOnDisk.read
        """
        # just create the variable and dimensions
        ds = self._ds
        dima = values # internal convention: values is a numpy array
        values, nctype = _maybe_convert_dtype(dima)

        assert self._name in ds.variables.keys(), "variable does not exist, should have been created earlier!"

        # add attributes
        if hasattr(dima,'attrs'):
            self.attrs.update(dima.attrs)

        # special case: index == slice(None) and self.ndim == 0
        # This would fail with numpy, but not with netCDF4
        if type(indices) is slice and indices == slice(None) and self.ndim == 0:
            indices = ()

        # do all the indexing and assignment via IndexedArray class
        # ==> it will set the values via _setvalues_ortho below
        # super(DimArrayOnDisk)._setitem(self, indices, values, **kwargs)
        AbstractDimArray._setitem(self, indices, values, axis=axis, indexing=indexing, tol=tol)

    def read(self, indices=None, axis=0, indexing=None, tol=None, keepdims=False):
        """ Read values from disk

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
        tol : float, optional
            tolerance when looking for numerical values, e.g. to use nearest 
            neighbor search, default `None`.
        keepdims : bool, optional 
            keep singleton dimensions (default False)

        Returns
        -------
        DimArray instance or scalar

        See Also
        --------
        DimArray.take, DatasetOnDisk.read, DimArrayOnDisk.write
        """
        if type(indices) is slice and indices == slice(None) and self.ndim == 0:
            indices = ()
        return AbstractDimArray._getitem(self, indices=indices, axis=axis, 
                                         indexing=indexing, tol=tol, keepdims=keepdims)
                           
    __setitem__ = write
    __getitem__ = read

    def _repr(self, **kwargs):
        assert 'lazy' not in kwargs, "lazy parameter cannot be provided, it is always True"
        return repr_dimarray(self, lazy=True, **kwargs)

    def _setvalues_ortho(self, idx_tuple, values, cast=False):
        if cast is True:
            warnings.warn("`cast` parameter is ignored")
        # values = _maybe_convert_dtype_array(values)
        values, nctype = _maybe_convert_dtype(values)
        self.values[idx_tuple] = values

    def _getvalues_ortho(self, idx_tuple):
        res = self.values[idx_tuple]
        # scalar become arrays with netCDF4# scalar become arrays with netCDF4
        # need convert to ndim=0 numpy array for consistency with axes
        if self.ndim == 0:
            try:
                res[0] + 1 # pb arises only for numerical types
                res = np.array(res[0]) 
            except:
                res = np.array(res) # str and unicode
        return res

    def _getaxes_ortho(self, idx_tuple):
        " idx: tuple of position indices  of length = ndim (orthogonal indexing)"
        axes = []
        for i, ix in enumerate(idx_tuple):
            ax = self.axes[i][ix]
            if not np.isscalar(ax): # do not include scalar axes
                axes.append(ax)
        return axes

class AxisOnDisk(GetSetDelAttrMixin, NetCDFVariable, AbstractAxis):
    def __init__(self, ds, name):
        self._ds = ds
        if not isinstance(name, basestring):
            raise TypeError("only string names allowed")
        self._name = name

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, new):
        self._ds.renameDimension(self._name, new)
        if self._name in self._ds.variables.keys():
            self._ds.renameVariable(self._name, new)
        self._name = new

    def __setitem__(self, indices, ax):
        ds = self._ds
        name = self._name

        assert getattr(ax, 'name', name) == name, "inconsistent axis name"

        values, nctype = _maybe_convert_dtype(ax)

        # assign value to variable (variable creation handled in AxesOnDisk)
        ds.variables[name][indices] = values

        # add attributes
        attrs = getattr(ax, 'attrs', {})
        try:
            self.attrs.update(attrs)
        except Exception as error:
            warnings.warn(error)

    def __getitem__(self, indices):
        name = self._name
        ds = self._ds
        if name in ds.variables.keys():
            # assume that the variable and dimension have the same name
            values = ds.variables[name][indices]
        else:
            # default, dummy dimension axis
            msg = "'{}' dimension not found, define integer range".format(name)
            warnings.warn(msg)
            values = np.arange(len(ds.dimensions[name]))[indices]

        # do not produce an Axis object
        if np.isscalar(values):
            return values

        ax = Axis(values, name)
        
        # add metadata
        if name in ds.variables.keys():
            ax.attrs.update(self.attrs)

        return ax

    def __len__(self):
        return len(self._ds.dimensions[self._name])

    @property
    def size(self):
        return len(self)

    def _bounds(self):
        size = len(self) # does not work with unlimited dimensions?
        if self.size == 0:
            first, last = None, None
        elif self._name in self._ds.variables.keys():
            first, last = self._ds.variables[self._name][0], self._ds.variables[self._name][size-1]
        else:
            first, last = 0, size-1
        return first, last

class AxesOnDisk(AbstractAxes):
    def __init__(self, ds, dims, **kwargs):
        self._ds = ds
        self._dims = dims
        self._kwargs = kwargs # for variable creation

    @property
    def dims(self):
        return self._dims

    def __len__(self):
        return len(self._dims)

    def __getitem__(self, dim):
        if not isinstance(dim, basestring):
            dim = self.dims[dim]
        elif dim not in self.dims:
            msg = "{} not found in dimensions (dims={}).".format(dim, self.dims)
            print """An error ocurred? A new dimension can be created via `axes.append(axis)` syntax, 
where axis can be a string (unlimited dimension), an Axis instance, 
or a tuple (name, values).  See help on `axes.append` for more information. 
Low-level netCDF4 function is also available as `ds.nc.createDimension` 
and `ds.nc.createVariable`"""
            raise KeyError(msg)
        return AxisOnDisk(self._ds, dim)

    def __setitem__(self, dim, ax):
        " modify existing axis and possibly create new associated variable " 
        if not isinstance(dim, basestring):
            dim = self.dims[dim]

        if dim not in self._ds.dimensions.keys():
            raise ValueError("{} dimension does not exist. Use axes.append() to create a new axis".format(dim))

        values, nctype = _maybe_convert_dtype(ax)

        if dim not in self._ds.variables.keys(): 
            v = self._ds.createVariable(dim, nctype, dim, **self._kwargs) 

        self[dim][:] = ax
        # v[:] = values

    def append(self, ax, **kwargs):
        """ create new dimension
        
        Parameters
        ----------
        ax : Axis or (name, values) tuple or str
            if str, create an Unlimited dimension
        **kwargs : passed to createVariable
        """
        if isinstance(ax, basestring):
            self._ds.createDimension(ax, None)
            self._dims += (ax,) 

        else:
            try:
                ax = Axis.as_axis(ax)
            except:
                raise TypeError("can only append Axis instances or (name, values),\
                                or provide str to create unlimited dimension")
            # elif isinstance(ax, Axis):
            self._ds.createDimension(ax.name, ax.size)
            self._dims += (ax.name,) 

            self._kwargs.update(kwargs) # createVariable parameters
            self[ax.name] = ax # assign values and attributes

    def __iter__(self):
        for dim in self.dims:
            yield AxisOnDisk(self._ds, dim)

    _repr = repr_axes

class AttrsOnDisk(object):
    """ represent netCDF Dataset or Variable Attribute
    """
    def __init__(self, obj):
        self.obj = obj
    def __setitem__(self, name, value):
        self.obj.setncattr(name, value)
    def __getitem__(self, name):
        return self.obj.getncattr(name)
    def __delitem__(self, name):
        return self.obj.delncattr(name)
    def update(self, attrs):
        for k in attrs.keys():
            self[k] = attrs[k]
    def keys(self):
        return self.obj.ncattrs()
    def values(self):
        return [self.obj.getncattr(k) for k in self.obj.ncattrs()]
    def todict(self):
        return odict(zip(self.keys(), self.values()))
    def __iter__(self):
        for k in self.keys():
            yield k
    def __len__(self):
        return len(self.keys())
    def __repr__(self):
        return repr_attrs(self)

###################################################
# Wrappers
###################################################


def open_nc(file_name, *args, **kwargs):
    """ open a netCDF file a la netCDF4, for interactive access to its properties

    Parameters
    ----------
    file_name : netCDF file name
    *args, **kwargs : passed to netCDF4.Dataset

    Returns
    -------
    dimarray.io.nc.DatasetOnDisk class (see help on this class for more info)

    Examples
    --------
    >>> ncfile = da.get_ncfile('greenland_velocity.nc')
    >>> ds = da.open_nc(ncfile)

    Informative Display similar to a in-memory Dataset

    >>> ds
    DatasetOnDisk of 6 variables (NETCDF4)
    0 / y1 (113): -3400000.0 to -600000.0
    1 / x1 (61): -800000.0 to 700000.0
    surfvelmag: (u'y1', u'x1')
    lat: (u'y1', u'x1')
    lon: (u'y1', u'x1')
    surfvely: (u'y1', u'x1')
    surfvelx: (u'y1', u'x1')
    mapping: nan

    Load variables with [:] syntax, like netCDF4 package

    >>> ds['surfvelmag'][:] # load one variable
    dimarray: 6893 non-null elements (0 null)
    0 / y1 (113): -3400000.0 to -600000.0
    1 / x1 (61): -800000.0 to 700000.0
    array(...)

    Indexing is similar to DimArray, and includes the sel, isel methods

    >>> ds['surfvelmag'].ix[:10, -1] # load first 10 y1 values, and last x1 value
    dimarray: 10 non-null elements (0 null)
    0 / y1 (10): -3400000.0 to -3175000.0
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)

    >>> ds['surfvelmag'].sel(x1=700000, y1=-3400000)
    0.0

    >>> ds['surfvelmag'].isel(x1=-1, y1=0)
    0.0

    Need to close the Dataset at the end

    >>> ds.close() # close

    Also usable as context manager

    >>> with da.open_nc(ncfile) as ds:
    ...     dataset = ds.read() # load full data set, same as da.read_nc(ncfile)
    >>> dataset
    Dataset of 6 variables
    0 / y1 (113): -3400000.0 to -600000.0
    1 / x1 (61): -800000.0 to 700000.0
    surfvelmag: (u'y1', u'x1')
    lat: (u'y1', u'x1')
    lon: (u'y1', u'x1')
    surfvely: (u'y1', u'x1')
    surfvelx: (u'y1', u'x1')
    mapping: nan
    """
    return DatasetOnDisk(file_name, *args, **kwargs)

def read_nc(f, names=None, *args, **kwargs):
    """  Wrapper around DatasetOnDisk.read  
    
    Read one or several variables from one or several netCDF file

    Parameters
    ----------
    f : str or netCDF handle
        netCDF file to read from or regular expression

    names : None or list or str, optional
        variable name(s) to read
        default is None

    indices : int or list or slice (single-dimensional indices)
               or a tuple of those (multi-dimensional)
               or `dict` of { axis name : axis indices }
        Indices refer to Dataset axes. Any item that does not possess
        one of the dimensions will not be indexed along that dimension.
        For example, scalar items will be left unchanged whatever indices
        are provided.

    axis : 
        When reading multiple files and align==True :
            axis along which to join the dimarrays or datasets (if align is True)
        When reading one file (deprecated: use {name:values} notation instead):
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

    align : bool, optional
        if names is a list of files or a regular expression, pass align=True
        if the arrays from the various files have to be aligned prior to 
        concatenation. Similar to dimarray.stack and dimarray.stack_ds

        Only when reading multiple files

    axis : str, optional
        axis along which to join the dimarrays or datasets (if align is True)

        Only when reading multiple files and align==True

    keys : sequence, optional
        new axis values. If not provided, file names will be taken instead.
        It is always possible to use set_axis later.

        Only when reading multiple files and align==True

    Returns
    -------
    obj : DimArray or Dataset
        depending on whether a (single) variable name is passed as argument (names) or not

    See Also
    --------
    DatasetOnDisk.read, stack, concatenate, stack_ds, concatenate_ds,
    DimArray.write_nc, Dataset.write_nc

    Examples
    --------
    >>> import os
    >>> from dimarray import read_nc, get_datadir

    Single netCDF file

    >>> ncfile = os.path.join(get_datadir(), 'cmip5.CSIRO-Mk3-6-0.nc')

    >>> data = read_nc(ncfile)  # load full file
    >>> data
    Dataset of 2 variables
    0 / time (451): 1850 to 2300
    1 / scenario (5): u'historical' to u'rcp85'
    tsl: (u'time', u'scenario')
    temp: (u'time', u'scenario')
    >>> data = read_nc(ncfile,'temp') # only one variable
    >>> data = read_nc(ncfile,'temp', indices={"time":slice(2000,2100), "scenario":"rcp45"})  # load only a chunck of the data
    >>> data = read_nc(ncfile,'temp', indices={"time":1950.3}, tol=0.5)  #  approximate matching, adjust tolerance
    >>> data = read_nc(ncfile,'temp', indices={"time":-1}, indexing='position')  #  integer position indexing

    Multiple files
    Read variable 'temp' across multiple files (representing various climate models)
    In this case the variable is a time series, whose length may 
    vary across experiments (thus align=True is passed to reindex axes before stacking)

    >>> direc = get_datadir()
    >>> temp = da.read_nc(direc+'/cmip5.*.nc', 'temp', align=True, axis='model')

    A new 'model' axis is created labeled with file names. It is then 
    possible to rename it more appropriately, e.g. keeping only the part
    directly relevant to identify the experiment:
    
    >>> getmodel = lambda x: os.path.basename(x).split('.')[1] # extract model name from path
    >>> temp.set_axis(getmodel, axis='model', inplace=True) # would return a copy if inplace is not specified
    >>> temp
    dimarray: 9114 non-null elements (6671 null)
    0 / model (7): 'CSIRO-Mk3-6-0' to 'MPI-ESM-MR'
    1 / time (451): 1850 to 2300
    2 / scenario (5): u'historical' to u'rcp85'
    array(...)
    
    This works on datasets as well:

    >>> ds = da.read_nc(direc+'/cmip5.*.nc', align=True, axis='model')
    >>> ds.set_axis(getmodel, axis='model')
    Dataset of 2 variables
    0 / model (7): 'CSIRO-Mk3-6-0' to 'MPI-ESM-MR'
    1 / time (451): 1850 to 2300
    2 / scenario (5): u'historical' to u'rcp85'
    tsl: ('model', u'time', u'scenario')
    temp: ('model', u'time', u'scenario')
    """
    # check for regular expression
    if type(f) is str:
        test = glob.glob(f)
        if len(test) == 1:
            f = test[0]
        elif len(test) == 0:
            raise ValueError('File is not present: '+repr(f))
        else:
            f = test
            f.sort()

    # multi-file ?
    if type(f) is list:
        mf = True
    else:
        mf = False

    # Read a single file
    if not mf:
        f, close = _maybe_open_file(f, mode='r')
        obj = DatasetOnDisk(f).read(names, *args, **kwargs)
        if close: f.close()

    # Read multiple files
    else:
        # single variable ==> DimArray (via Dataset)
        if names is not None and isinstance(names, str):
            obj = _read_multinc(f, [names], *args, **kwargs)
            obj = obj[names]

        # single variable ==> DimArray
        else:
            obj = _read_multinc(f, names, *args, **kwargs)

    return obj

def _read_multinc(fnames, names=None, axis=None, keys=None, align=False, concatenate_only=False, **kwargs):
    """ read multiple netCDF files 

    Parameters
    ----------
    fnames : list of file names or file handles to be read
    names : variable names to be read
    axis : str, optional
        dimension along which the files are concatenated 
        (created as new dimension if not already existing)
    keys : sequence, optional
        to be passed to stack_nc, if axis is not part of the dataset
    align : `bool`, optional
        if True, reindex axis prior to stacking (default to False)
    concatenate_only : `bool`, optional
        if True, only concatenate along existing axis (and raise error if axis not existing) 

    **kwargs : keyword arguments passed to DatasetOnDisk.read  (cannot 
    contain 'axis', though, but indices can be passed as a dictionary
    if needed, e.g. {'time':2010})

    Returns
    -------
    dimarray's Dataset instance

    This function reads several files and call stack_ds or concatenate_ds 
    depending on whether `axis` is absent from or present in the dataset, respectively

    See `dimarray.read_nc` for more complete documentation.

    See Also
    --------
    read_nc
    """
    variables = None
    dimensions = None

    datasets = []
    for fn in fnames:
        with DatasetOnDisk(fn) as f:
            ds = f.read(names, **kwargs)

        # check that the same variables are present in the file
        if variables is None:
            variables = ds.keys()
        else:
            variables_new = ds.keys()
            assert variables == variables_new, \
                    "netCDF files must contain the same \
                    subset of variables to be concatenated/stacked"

        # check that the same dimensions are present in the file
        if dimensions is None:
            dimensions = ds.dims
        else:
            dimensions_new = ds.dims
            assert dimensions == dimensions_new, \
                    "netCDF files must contain the same \
                    subset of dimensions to be concatenated/stacked"

        datasets.append(ds)

    # check that dimension in dataset if required
    if concatenate_only and axis not in dimensions:
        raise Exception('required axis {} not found, only got {}'.format(axis, dimensions))

    # Join dataset
    if axis in dimensions:
        if keys is not None: warnings.warn('keys argument will be ignored.')
        ds = concatenate_ds(datasets, axis=axis)

    else:
        # use file name as keys by default
        if keys is None:
            keys = [os.path.splitext(fn)[0] for fn in fnames]
        ds = stack_ds(datasets, axis=axis, keys=keys, align=align)

    return ds

#
# display summary information
#

def summary_nc(fname, name=None, metadata=False):
    """ Print summary information about the content of a netCDF file
    Deprecated, see dimarray.open_nc
    """
    warnings.warn("Deprecated. Use dimarray.open_nc", FutureWarning)
    with DatasetOnDisk(fname) as obj:
        if name is not None:
            obj = obj[name]
        print(obj.__repr__(metadata=metadata))


def _maybe_open_file(f, mode='r', clobber=None, verbose=False, format=None):
    """ open a netCDF4 file

    Parameters
    ----------
    f : file name (str) or netCDF file handle
    mode: changed from original 'r','w','r' & clobber option:

    mode : `str`
        read or write access
        - 'r': read 
        - 'w' : write, overwrite if file if present (clobber=True)
        - 'w-': create new file, but raise Exception if file is present (clobber=False)
        - 'a' : append, raise Exception if file is not present
        - 'a+': append if file is present, otherwise create

    format: passed to netCDF4.Dataset, only relevatn when mode = 'w', 'w-', 'a+'
        'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT'
         
    Returns
    -------
    f : netCDF file handle
    close: `bool`, `True` if input f indicated file name
    """
    format = format or get_option('io.nc.format')

    if mode == 'w-':
        mode = 'w'
        if clobber is None: clobber = False

    # mode 'a+' appends if file exists, otherwise create new variable
    elif mode == 'a+' and not isinstance(f, nc.Dataset):
        if os.path.exists(f): mode = 'a'
        else: mode = 'w'
        if clobber is None: clobber=False

    else:
        if clobber is None: clobber=True

    if not isinstance(f, nc.Dataset):
        fname = f

        # make sure the file does not exist if mode == "w"
        if os.path.exists(fname) and clobber and mode == "w":
            os.remove(fname)

        try:
            f = nc.Dataset(fname, mode, clobber=clobber, format=format)
        except UserWarning, msg:
            print msg
        except Exception, msg: # raise a weird RuntimeError
            #print "read from",fname
            raise IOError("{} => failed to opend {} in mode {}".format(msg, fname, mode)) # easier to handle

        if verbose: 
            if 'r' in mode:
                print "read from",fname
            elif 'a' in mode:
                print "append to",fname
            else:
                print "write to",fname
        close = True

    return f, close
