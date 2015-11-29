""" NetCDF I/O to access and save geospatial data
"""
import os
import re
import glob, copy
from collections import OrderedDict as odict
from functools import partial
import warnings
import numpy as np
import netCDF4 as nc
import dimarray as da
#from geo.data.index import to_slice, _slice3D

from dimarray.compat.pycompat import basestring, zip
from dimarray.tools import format_doc
from dimarray.dataset import Dataset, concatenate_ds, stack_ds
from dimarray.core import DimArray, Axis, Axes
from dimarray.config import get_option
from dimarray.core.bases import AbstractDimArray, AbstractDataset, AbstractAxis, GetSetDelAttrMixin, AbstractAxes
# from dimarray.core.metadata import _repr_metadata
from dimarray.prettyprinting import repr_axis, repr_axes, repr_dimarray, repr_dataset, repr_attrs

from .conventions import encode_cf_datetime, decode_cf_datetime

__all__ = ['read_nc','summary_nc', 'write_nc']


#
# Global variables 
#
FORMAT = get_option('io.nc.format') # for the doc

#
# Helper functions
#
def maybe_encode_values(values, format=None):
    """ strings are given "object" type in Axis object
    ==> assume all objects are actually strings
    NOTE: this will fail for other object-typed axes such as tuples
    """
    # if dtype is np.dtype('O'):
    values = np.asarray(values)
    dtype = values.dtype
    cf_attrs = {}

    if dtype.kind in ('S','O'):
        encoded = np.asarray(values, dtype=object)
        dtype = str
    elif dtype.kind == 'M': # 'datetime64'
        # values = np.asarray(np.datetime_as_string(values), dtype=object)
        # dtype = str
        encoded, units, calendar = encode_cf_datetime(values)
        dtype = encoded.dtype
        cf_attrs['calendar'] = calendar
        cf_attrs['units'] = units
    elif dtype.kind == 'm': # 'timedelta64'
    # elif np.issubdtype(dtype, np.timedelta64):
        encoded, units = encode_cf_timedelta(values)
        cf_attrs['units'] = units
        dtype = encoded.dtype
    else:
        encoded = values

    # if the format is NETCDF3, uses int32 instead of int64
    if format == "NETCDF3_CLASSIC" and dtype is np.dtype("int64"):
        warnings.warn("convert int64 into int32 for writing to NETCDF3 format", RuntimeWarning)
        dtype = "int32"
    elif format == "NETCDF3_64BIT" and dtype is np.dtype("int64"):
        # it is strange that a warnings also occurs in the _64BIT version !
        warnings.warn("convert int64 into int32 for writing to NETCDF3 format", RuntimeWarning)
        dtype = "int32"
    return encoded, dtype, cf_attrs

# quick and dirty conversion from string
TIME_UNITS = ['years','months','days','hours','minutes', 'seconds']

def hastimeunits(ncvar):
    regexpr = "{} +since".format("|".join(TIME_UNITS))
    return hasattr(ncvar, 'units') and re.match(regexpr, ncvar.units)

def istimevariable(ncvar):
    """Return True if a netCDF4 variable represents time
    """
    # return hastimeunits(ncvar) or \
    return (
        ncvar.size > 0 and isinstance(ncvar[0], basestring) \
         and re.match('\d\d(\d\d)?-\d\d-\d\d',ncvar[0]))

class NCTimeVariableWrapper(object):
    """Subclass of netCDF4 variable that performs conversions
    to datetime64 object when indexing. Should only be used for 
    object or string types
    """
    def __init__(self, ncvar):
        self.ncvar = ncvar
    def __getitem__(self, idx):
        arr = self.ncvar[idx]
        if arr.dtype.kind in ('S','O'):
            try:
                arr = np.asarray(arr, dtype='datetime64')
            except Exception as error:
                warnings.warn("Failed to convert string time to datetime64\n"+str(error), RuntimeWarning)
        else:
            try:
                arr = decode_cf_datetime(arr, getattr(self.ncvar, 'units',None), getattr(self.ncvar, 'calendar', None))
            except Exception as error:
                warnings.warn("Failed to convert {} to datetime64\n".format(arr.dtype)+str(error), RuntimeWarning)
        return arr
    def __getattr__(self, att):
        return getattr(self.ncvar, att)

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
            try:
                self._ds = nc.Dataset(f, mode=mode, clobber=clobber, diskless=diskless, persist=persist, format=format)
            except UserWarning as error:
                print error
            except Exception as error: # indicate file name when issuing error
                raise IOError("{}\n=> failed to open {} (mode={}, clobber={})".format(str(error), f, mode, clobber)) # easier to handle
        # assert self._kwargs is not None
        self._kwargs = kwargs
    @property
    def _obj(self):
        return self._ds

    def read(self, names=None, indices=None, axis=0, indexing=None, tol=None, keepdims=False,
             verbose=False, # back-compatibility
             ):
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
        if verbose:
            # print "Read ",self.filename
            pass
        # automatically read all variables to load (except for the dimensions)
        if names is None:
            dims = self.dims
            names = self.keys()
        elif isinstance(names, basestring):
            return self[names].read(indices=indices, axis=axis, indexing=indexing, tol=tol, keepdims=keepdims)
        else:
            dims = []
        # else:
        #     raise TypeError("Expected list or str for 'names=', got {}".format(names))

        tuple_indices = self._get_indices(indices, axis=axis, tol=tol, keepdims=keepdims, indexing=indexing)
        dict_indices = {dim:tuple_indices[i] for i, dim in enumerate(self.dims)}

        data = Dataset()

        # first load dimensions
        for dim in dims:
            data.axes.append(self.axes[dim][dict_indices[dim]])

        # then normal variables
        for nm in names:
            data[nm] = self[nm].read(indices={dim:dict_indices[dim] for dim in self[nm].dims}, indexing='position')
        data.attrs.update(self.attrs) # dataset's metadata

        # reorder the axes in the dataset to match input
        data.axes = Axes(data.axes[dim] for dim in self.dims if dim in data.dims)

        return data

    # so that loc, iloc, sel, isel, nloc work:
    _getitem = read 

    def __getitem__(self, name):
        if name in self.dims:
            return DimArrayOnDisk(self._ds, name)
            # raise KeyError("Use 'axes' property to access dimension variables: {}".format(name))
            # return AxisOnDisk(self._ds, name)

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
        _, nctype, cf_attrs = maybe_encode_values(dima, format=self._ds.file_format)

        name = name or getattr(self, "name", None)
        if not name:
            raise ValueError("Need to provide variable name")

        kw = self._kwargs.copy() 
        kw.update(kwargs)

        if name not in self._ds.variables.keys():

            if np.isscalar(dima):
                dima = da.DimArray(dima)
            if not isinstance(dima, DimArray):
                raise TypeError("Expected DimArray, got {}".format(type(dima)))
            if dima.ndim > 0:
                # create Dimension and associated variable
                for ax in dima.axes:
                    if ax.name not in self.dims:
                        self.axes.append(ax, **kw)

            # add _FillValue or missing_value attribute to fill_value
            if 'fill_value' not in kw and hasattr(dima, '_FillValue'):
                kw['fill_value'] = dima._FillValue
            elif 'fill_value' not in kw and hasattr(dima, 'missing_value'):
                kw['fill_value'] = dima.missing_value

            # if the variable is already present (e.g. as present as variable 
            # AND dimension in dimarray Dataset), just skip it
            if  name in self._ds.variables.keys():
                pass
                # warnings.warn("dimension variable present both as variable and Axis in DimArra", RuntimeWarning)
            else:
                self._ds.createVariable(name, nctype, dima.dims, **kw)

        dimaondisk = DimArrayOnDisk(self._ds, name)
        dimaondisk[()] = dima
        dimaondisk.attrs.update(cf_attrs) # calendar?

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
    @dims.setter
    def dims(self, newdims):
        self._set_dims(newdims)

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
    @property
    def __array__(self):
        return self.values[()].__array__ # returns a numpy array
    
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
        values, nctype, cf_attrs = maybe_encode_values(values, format=self._ds.file_format)

        assert self._name in ds.variables.keys(), "variable does not exist, should have been created earlier!"

        # add attributes
        if hasattr(dima,'attrs'):
            self.attrs.update(dima.attrs)
            self.attrs.update(cf_attrs) # calendar?

        # special case: index == slice(None) and self.ndim == 0
        # This would fail with numpy, but not with netCDF4
        if type(indices) is slice and indices == slice(None) and self.ndim == 0:
            indices = ()

        indices = self._get_indices(indices,axis=axis, indexing=indexing, tol=tol)

        # Perform additional checks on axes if the Data to assign is a DimArray
        if isinstance(dima, DimArray):
            for i, ax in enumerate(self.axes):
                idx = indices[i]
                axis = dima.axes[ax.name]
                # write unlimited dimensions
                if self._ds.dimensions[ax.name].isunlimited():
                    self.axes[ax.name][idx] = axis
                else:
                    # dimension variable already written, simple check
                    ondisk = self.axes[ax.name][idx if not np.isscalar(idx) else [idx]].values
                    inmemory = axis.values
                    # inmemory, _ = maybe_encode_values(axis.values)
                    if not np.all(ondisk == inmemory):
                        # assert np.all(ondisk == inmemory)
                        warnings.warn("axes values differ in the netCDF file, try using a numpy array instead, or re-index\
                                      the dimarray prior to assigning to netCDF", RuntimeWarning)

        # do all the indexing and assignment via IndexedArray class
        # ==> it will set the values via _setvalues_ortho below
        # super(DimArrayOnDisk)._setitem(self, indices, values, **kwargs)
        AbstractDimArray._setitem(self, indices, values, indexing='position')

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

    # so that loc, iloc, sel, isel, nloc work:
    _getitem = read
    _setitem = write

    def _repr(self, **kwargs):
        assert 'lazy' not in kwargs, "lazy parameter cannot be provided, it is always True"
        return repr_dimarray(self, lazy=True, **kwargs)

    def _setvalues_ortho(self, idx_tuple, values, cast=False):
        if cast is True:
            warnings.warn("`cast` parameter is ignored", DeprecationWarning)
        # values = maybe_encode_values(values)
        values, nctype, cf_attrs = maybe_encode_values(values)
        self.values[idx_tuple] = values
        self.attrs.update(cf_attrs)

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

    def _isdefined(self):
        return self._name in self._ds.variables
    def _ncvar(self):
        return self._ds.variables[self._name]
    def _ncdim(self):
        return self._ds.dimensions[self._name]
    def _istime(self):
        return self._isdefined() and istimevariable(self._ncvar())

    @property
    def values(self):
        if self._isdefined():
            ncvar = self._ncvar()
            if self._istime():
                values = NCTimeVariableWrapper(ncvar)
            else:
                values = ncvar
        else:
            # default, dummy dimension axis
            msg = "'{}' dimension not found, define integer range".format(self._name)
            warnings.warn(msg, RuntimeWarning)
            values = np.arange(len(self._ncdim()))
        return values

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, new):
        self._ds.renameDimension(self._name, new)
        if self._name in self._ds.variables.keys():
            self._ds.renameVariable(self._name, new)
        self._name = new

    @property
    def attrs(self):
        if self._istime():
            return AttrsOnDisk(self._obj, exclude=['calendar','units'])
        else:
            return AttrsOnDisk(self._obj)

    def __setitem__(self, indices, ax):
        ds = self._ds
        name = self._name

        assert getattr(ax, 'name', name) == name, "inconsistent axis name"

        values, nctype, cf_attrs = maybe_encode_values(ax, format=self._ds.file_format)

        # assign value to variable (finer-grained control in AxesOnDisk.append)
        if name not in self._ds.variables.keys(): 
            ds.createVariable(name, nctype, name)

        if np.isscalar(indices): 
            assert values.size == 1
            if values.ndim == 0:
                values = values[()]
            else:
                values = values[0]

        ds.variables[name][indices] = values

        # add attributes
        attrs = getattr(ax, 'attrs', {})
        try:
            self.attrs.update(attrs)
        except Exception as error:
            warnings.warn(str(error), RuntimeWarning)
        self.attrs.update(cf_attrs)

    def __getitem__(self, indices):
        values = self.values[indices]

        # do not produce an Axis object
        if np.isscalar(values):
            return values

        if isinstance(values, np.ma.MaskedArray):
            values = values.filled(0)

        ax = Axis(values, self._name)

        # add metadata
        if self._name in self._ds.variables.keys():
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

    @property
    def dtype(self):
        if self._name in self._ds.variables.keys():
            return np.dtype(self._ds.variables[self._name].dtype)
        else:
            return np.dtype('i')

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

        values, nctype, cf_attrs = maybe_encode_values(ax, format=self._ds.file_format)

        if dim not in self._ds.variables.keys(): 
            v = self._ds.createVariable(dim, nctype, dim, **self._kwargs) 

        self[dim][:] = ax
        self[dim].attrs.update(cf_attrs)
        # v[:] = values

    def append(self, ax, size=None, **kwargs):
        """ create new dimension
        
        Parameters
        ----------
        ax : str or Axis or (name, values) tuple
            if str, simply create a dimension without writing the axis values
        size : int or None, optional
            if None, an unlimited dimension is created
            if ax is provided as an array-like or Axis, size is taken 
            from the axis values by default (so size does not need to 
            be provided)
        **kwargs : passed to createVariable (compression parameters)
        """
        if isinstance(ax, basestring):
            self._ds.createDimension(ax, size)
            self._dims += (ax,) 

        else:
            try:
                ax = Axis.as_axis(ax)
            except:
                raise TypeError("can only append Axis instances or (name, values),\
                                or provide str to create unlimited dimension")
            # elif isinstance(ax, Axis):
            size = size or ax.size
            self._ds.createDimension(ax.name, size)
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
    def __init__(self, obj, exclude=None):
        self.obj = obj
        self.exclude = exclude
    def __setitem__(self, name, value):
        if name == "_FillValue": return
        try:
            self.obj.setncattr(name, value)
        except TypeError as error:
            warnings.warn(str(error), RuntimeWarning)
            if type(value) is bool:
                warnings.warn("convert boolean to integer", RuntimeWarning)
                self.obj.setncattr(name, int(value))

    def __getitem__(self, name):
        return self.obj.getncattr(name)
    def __delitem__(self, name):
        return self.obj.delncattr(name)
    def update(self, attrs):
        for k in attrs.keys():
            self[k] = attrs[k]
    def keys(self):
        keys = getattr(self.obj,'ncattrs', lambda :[])()
        if self.exclude is not None:
            keys = [k for k in keys if k not in self.exclude]
        return keys
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

    >>> ds    # doctest: +SKIP
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

    axis : str, optional
        When reading multiple files, axis along which to join
        the dimarrays or datasets. It the axis already exist, 
        the resulting arrays will be concatenated, otherwise
        they will be stacked along a new array (in the sense 
        of the numpy functions `concatenate` and `stack`)

    keys : sequence, optional
        When reading multiple files, keys for the join axis. If the 
        axis already exists in the dataset, the concatenated dataset/dimarray
        will be re-indexed along the provided key, otherwise the keys
        will be used to create a new axis for stacking. In the latter case,
        keys' length needs to exactly match the number of input files, and 
        if not provided, file names will be taken instead. Note you may
        manually rename the axes later, or use the `set_axis` method.

    align : bool, optional
        When reading multiple files, passed to `stack` (new axis) or 
        `concatenate` (existing axis) to reindex all arrays onto common axes.
        (in `concatenate` mode, the concatenation axis is *not* re-indexed of course, 
        only the secondary axes)
        Default to False.

    **kwargs : optional key-word arguments passed to align, if align is True
        When reading multiple files, passed to `stack` (new axis) or 
        This includes: `sort` (False by default) and `join` ('outer' by default)

    Returns
    -------
    obj : DimArray or Dataset
        depending on whether a (single) variable name is passed as argument (names) or not

    See Also
    --------
    DatasetOnDisk.read, stack, concatenate, stack_ds, concatenate_ds, align,
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
    >>> temp.set_axis(getmodel, axis='model') # would return a copy if inplace is not specified
    >>> temp
    dimarray: 9114 non-null elements (6671 null)
    0 / model (7): 'CSIRO-Mk3-6-0' to 'MPI-ESM-MR'
    1 / time (451): 1850 to 2300
    2 / scenario (5): u'historical' to u'rcp85'
    array(...)
    
    This works on datasets as well:

    >>> ds = da.read_nc(direc+'/cmip5.*.nc', align=True, axis='model')
    >>> ds.set_axis(getmodel, axis='model')
    >>> ds
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

def _read_multinc(fnames, names=None, axis=None, keys=None, align=False, sort=False, join='outer', concatenate_only=False, **kwargs):
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
        if True, reindex axis prior to stacking / concatenating (default to False)
    sort : `bool`, optional
        if True, sort common axes prior to stacking / concatenating (defaut to False)
    join : `str`, optional
        join method in align (default to 'outer')
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
        ds = concatenate_ds(datasets, axis=axis, align=align, sort=sort, join=join)
        if keys is not None: 
            ds = ds.reindex_axis(keys, axis=axis)
        # elif sort:
        #     ds = ds.sort_axis(axis=axis)
            #warnings.warn('keys argument will be ignored.', RuntimeWarning)

    else:
        # use file name as keys by default
        if keys is None:
            keys = [os.path.splitext(fn)[0] for fn in fnames]
        ds = stack_ds(datasets, axis=axis, keys=keys, align=align, sort=sort, join=join)

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
    else:
        close = False # leave it open

    return f, close
