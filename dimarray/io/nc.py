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
def _maybe_convert_dtype(values, f=None):
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

# def _maybe_convert_dtype_array(arr):
#     arr = np.asarray(arr)
#     if arr.dtype > np.dtype('S1'):
#         arr = np.asarray(arr, dtype=object)
#     return arr

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
    # @staticmethod
    # def _convert_dtype(dtype):
    #     # needed for VLEN types
    #     # ==> assume all objects are actually strings
    #     #NOTE: this will fail for other object-typed axes such as tuples
    #     if dtype >= np.dtype('S1'): # all strings, this include objects dtype('O')
    #         nctype = str 
    #     else:
    #         nctype = dtype 
    #     return nctype


class DatasetOnDisk(GetSetDelAttrMixin, NetCDFOnDisk, AbstractDataset):
    def __init__(self, f, *args, **kwargs):
        if isinstance(f, nc.Dataset):
            self._ds = f
        else:
            self._ds = nc.Dataset(f, *args, **kwargs)
    @property
    def _obj(self):
        return self._ds

    def read(self, names=None, **kwargs):
        """ Equivalent to dimarray.read_nc
        """
        # automatically read all variables to load (except for the dimensions)
        if names is None:
            names = self.keys()
        data = Dataset()
        for nm in names:
            data[nm] = self[nm].read(**kwargs)
        data.attrs.update(self.attrs) # dataset's metadata
        return data

    def write(self, obj, *args, **kwargs):
        if not isinstance(obj, da.Dataset):
            raise TypeError("Can only write Dataset, use `ds[name] = dima` to write a DimArray")
        for k in obj.keys():
            self[k].write(obj.axes, obj.values)

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
        return AxesOnDisk(self._ds, self.dims)

    def __getitem__(self, name):
        if name in self.dims:
            raise KeyError("Use 'axes' property to access dimension variables: {}".format(name))
        if name not in self.keys():
            raise KeyError("Variable not found in the dataset: {}.\nExisting variables: {}".format(name, self.keys()))
        return DimArrayOnDisk(self._ds, name)

    def __setitem__(self, name, value):
        DimArrayOnDisk(self._ds, name)[:] = value

    def __delitem__(self, name):
        del self._ds.variables[name]

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
        self[:] = values
    def __array__(self):
        return self.values[:] # returns a numpy array
    
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

    def write(self, indices, values, **kwargs):
        # just create the variable and dimensions
        ds = self._ds
        arr, nctype = _maybe_convert_dtype(values, ds)

        if self._name not in ds.variables.keys():
            # create variable
            if np.isscalar(values):
                values = da.DimArray(values)
            if not isinstance(values, DimArray):
                #TODO: do the scalar case
                raise TypeError("Can only create new variables with DimArray")
            if values.ndim > 0:
                for ax in values.axes:
                    AxisOnDisk(ds, ax.name)[:] = ax.values
            ds.createVariable(self._name, nctype, values.dims)

        # add attributes
        if hasattr(values,'attrs'):
            self.attrs.update(values.attrs)

        # do all the indexing and assignment via IndexedArray class
        # ==> it will set the values via _setvalues_ortho below
        # super(DimArrayOnDisk)._setitem(self, indices, values, **kwargs)
        AbstractDimArray._setitem(self, indices, arr, **kwargs)

    write.__doc__ = AbstractDimArray._setitem.__doc__

    def read(self, indices=slice(None), *args, **kwargs):
        return AbstractDimArray._getitem(self, indices, *args, **kwargs)
                           
    read.__doc__ = AbstractDimArray._getitem.__doc__
    # read = AbstractDimArray._getitem #TODO: wrap documentation

    __setitem__ = write
    __getitem__ = read

    def _repr(self, **kwargs):
        assert 'lazy' not in kwargs, "lazy parameter cannot be provided, it is always True"
        return repr_dimarray(self, lazy=True, **kwargs)

    def _setvalues_ortho(self, idx_tuple, values, cast=False):
        if cast is True:
            warnings.warn("`cast` parameter is ignored")
        # values = _maybe_convert_dtype_array(values)
        values, nctype = _maybe_convert_dtype(values, self._ds)
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
        if type(name) not in (str, unicode):
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

    def __setitem__(self, ix, values):
        ds = self._ds
        name = self._name

        size = np.size(values) if values is not None else None

        # create dimension variable if needed
        if name not in ds.dimensions.keys(): 
            ds.createDimension(name, size)

        # assign variable
        if values is None:
            return

        # arr = _maybe_convert_dtype_array(values)
        values, nctype = _maybe_convert_dtype(values, ds)

        # create variable if needed
        if name not in ds.variables.keys():
            # nctype = _maybe_convert_dtype(arr.dtype, ds)
            ds.createVariable(name, nctype, name) 

        # assign value to variable
        ds.variables[name][ix] = values

        # assign metadata
        try:
            self.attrs.update(values.attrs)
        except:
            pass

    def __getitem__(self, ix):
        name = self._name
        ds = self._ds
        if name in ds.variables.keys():
            # assume that the variable and dimension have the same name
            values = ds.variables[name][ix]
        else:
            # default, dummy dimension axis
            msg = "'{}' dimension not found, define integer range".format(name)
            warnings.warn(msg)
            values = np.arange(len(ds.dimensions[name]))[ix]

        # do not produce an Axis object
        if np.isscalar(values):
            return values

        axis = Axis(values, name)
        
        # add metadata
        if name in ds.variables.keys():
            axis.attrs.update(self.attrs)

        return axis

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

    _repr = repr_axis

class AxesOnDisk(AbstractAxes):
    def __init__(self, ds, dims):
        self._ds = ds
        self._dims = dims
    @property
    def dims(self):
        return self._dims
    def __getitem__(self, dim):
        if type(dim) is int:
            dim = self.dims[dim]
        elif dim not in self.dims:
            msg = """{} not found in dimensions (dims={}).
A new dimension can be created via: `ds.axes[name] = values` 
or `ds.axes[name] = None` for an unlimited dimension.
Low-level netCDF4 function is also available as `ds.nc.createDimension` 
and `ds.nc.createVariable`""".format(ix, self.dims)
            raise KeyError(msg)
        return AxisOnDisk(self._ds, dim)
    def __setitem__(self, dim, axis):
        if type(dim) is int:
            dim = self.dims[dim]
        # involves variable creation
        AxisOnDisk(self._ds, dim)[:] = axis

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
    Dataset of 6 variables (netCDF)
    0 / y1 (113): -3400000.0 to -600000.0
    1 / x1 (61): -800000.0 to 700000.0
    surfvelmag: (u'y1', u'x1')
    lat: (u'y1', u'x1')
    lon: (u'y1', u'x1')
    surfvely: (u'y1', u'x1')
    surfvelx: (u'y1', u'x1')
    mapping: ()

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

#
# some common piece of documentation
#
_doc_indexing = """
    indices : `dict`
        provide indices or slice to extract {nm1:val1}
    indexing : 'label' or 'position'
        By default, indexing is by axis values ('label') but 
        in some cases it is more convenient to provide it directly 
        by by position on the axis ('position'), in particular when 
        the first (0) or last (-1) values are requested.
    tol : None or float
        floating point tolerance when indexing float-arrays (default to None for exact match) 
""".strip()

_doc_indexing_write = """
    {take} 
""".format(take=_doc_indexing).strip()

_doc_write_nc = """ 
    format : `str`
        netCDF file format. Default is '{format}' (only accounted for during file creation)
    zlib : `bool`
        Enable zlib compression if True. Default is False (no compression).
    complevel : `int`
        integer between 1 and 9 describing the level of compression desired. Ignored if zlib=False.
    **kwargs : key-word arguments
        Any additional keyword arguments accepted by `netCDF4.Dataset.CreateVariable`
    
    Notes
    -----
    See the netCDF4-python module documentation for more information about the use
    of keyword arguments to write_nc.
""".strip().format(format=FORMAT)+'\n'

_doc_write_modes = """Several write modes are available:

        - 'w' : write, overwrite if file if present (clobber=True)
        - 'w-': create new file, but raise Exception if file is present (clobber=False)
        - 'a' : append, raise Exception if file is not present
        - 'a+': append if file is present, otherwise create
        """


@format_doc(indexing=_doc_indexing)
def read_nc(f, nms=None, *args, **kwargs):
    """  Read one or several variables from one or several netCDF file

    Parameters
    ----------
    f : str or netCDF handle
        netCDF file to read from or regular expression
    nms : None or list or str, optional
        variable name(s) to read
        default is None
    {indexing}

    align : bool, optional
        if nms is a list of files or a regular expression, pass align=True
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
        depending on whether a (single) variable name is passed as argument (nms) or not

    See Also
    --------
    summary_nc, take, stack, concatenate, stack_ds, concatenate_ds,
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
    1 / scenario (5): historical to rcp85
    tsl: (u'time', u'scenario')
    temp: (u'time', u'scenario')
    >>> data = read_nc(ncfile,'temp') # only one variable
    >>> data = read_nc(ncfile,'temp', indices={{"time":slice(2000,2100), "scenario":"rcp45"}})  # load only a chunck of the data
    >>> data = read_nc(ncfile,'temp', indices={{"time":1950.3}}, tol=0.5)  #  approximate matching, adjust tolerance
    >>> data = read_nc(ncfile,'temp', indices={{"time":-1}}, indexing='position')  #  integer position indexing

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
    0 / model (7): CSIRO-Mk3-6-0 to MPI-ESM-MR
    1 / time (451): 1850 to 2300
    2 / scenario (5): historical to rcp85
    array(...)
    
    This works on datasets as well:

    >>> ds = da.read_nc(direc+'/cmip5.*.nc', align=True, axis='model')
    >>> ds.set_axis(getmodel, axis='model')
    Dataset of 2 variables
    0 / model (7): CSIRO-Mk3-6-0 to MPI-ESM-MR
    1 / time (451): 1850 to 2300
    2 / scenario (5): historical to rcp85
    tsl: ('model', 'time', 'scenario')
    temp: ('model', 'time', 'scenario')
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

        # single variable ==> DimArray
        if nms is not None and (isinstance(nms, str) or type(nms) is unicode):
            obj = _read_variable(f, nms, *args, **kwargs)

        # multiple variable ==> Dataset
        else:
            obj = _read_dataset(f, nms, *args, **kwargs)

    # Read multiple files
    else:
        # single variable ==> DimArray (via Dataset)
        if nms is not None and isinstance(nms, str):
            obj = _read_multinc(f, [nms], *args, **kwargs)
            obj = obj[nms]

        # single variable ==> DimArray
        else:
            obj = _read_multinc(f, nms, *args, **kwargs)

    return obj


def _extract_kw(kwargs, argnames, delete=True):
    """ extract check 
    """
    kw = {}
    for k in kwargs.copy():
        if k in argnames:
            kw[k] = kwargs[k]
            if delete: del kwargs[k]
    return kw

@format_doc(indexing=_doc_indexing)
def _read_variable(f, name, indices=None, axis=0, indexing='label', tol=None, verbose=False):
    """ Read one variable from netCDF4 file 

    Parameters
    ----------
    f  : file name or file handle
    name : netCDF variable name to extract
    {indexing}

    Returns
    -------
    Returns a Dimarray instance

    See Also
    --------
    See preferred function `dimarray.read_nc` for more complete documentation 
    """
    f, close = _check_file(f, mode='r', verbose=verbose)
    obj = DimArrayOnDisk(f, name).read(indices, axis=axis, indexing=indexing, tol=tol)
    if close: f.close() # close netCDF if file was given as file name
    return obj

#read_nc.__doc__ += _read_variable.__doc__

def _read_dataset(f, nms=None, **kwargs):
    """ Read several (or all) variable from a netCDF file

    Parameters
    ----------
    f : file name or (netcdf) file handle
    nms : list of variables to read (default None for all variables)
    **kwargs

    Returns
    -------
    Dataset instance

    See preferred function `dimarray.read_nc` for more complete documentation 
    """
    kw = _extract_kw(kwargs, ('verbose',))
    f, close = _check_file(f, 'r', **kw)
    data = DatasetOnDisk(f).read(nms, **kwargs)
    if close: f.close()

    return data

def _read_multinc(fnames, nms=None, axis=None, keys=None, align=False, concatenate_only=False, **kwargs):
    """ read multiple netCDF files 

    Parameters
    ----------
    fnames : list of file names or file handles to be read
    nms : variable names to be read
    axis : str, optional
        dimension along which the files are concatenated 
        (created as new dimension if not already existing)
    keys : sequence, optional
        to be passed to stack_nc, if axis is not part of the dataset
    align : `bool`, optional
        if True, reindex axis prior to stacking (default to False)
    concatenate_only : `bool`, optional
        if True, only concatenate along existing axis (and raise error if axis not existing) 

    **kwargs : keyword arguments passed to io.nc._read_variable  (cannot 
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
        ds = _read_dataset(fn, nms, **kwargs)

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

@format_doc(netCDF4=_doc_write_nc, indexing=_doc_indexing_write, write_modes=_doc_write_modes)
def _write_dataset(f, obj, mode='w-', indices=None, axis=0, format=None, verbose=False, **kwargs):
    """ Write Dataset to netCDF file

    Parameters
    ----------
    f : str or netCDF handle
        netCDF file to write to
    mode : str, optional
        File creation mode. Default is 'w-'. Set to 'w' to overwrite any existing file.
        {write_modes}
    {netCDF4}
    {indexing}

    See Also
    --------
    read_nc
    DimArray.write_nc
    """
    f, close = _check_file(f, mode=mode, verbose=verbose, format=format)
    nms = obj.keys()
        
    for nm in obj:
        _write_variable(f, obj[nm], nm, **kwargs)

    # set metadata for the whole dataset
    meta = obj.attrs
    for k in meta.keys():
        if meta[k] is None: 
            # do not write empty attribute
            # but delete any existing attribute 
            # of the same name
            if k in f.ncattrs:
                f.delncattr(k) 
            continue
        f.setncattr(k, meta[k])

    if close: f.close()


@format_doc(netCDF4=_doc_write_nc, indexing=_doc_indexing_write, write_modes=_doc_write_modes)
def _write_variable(f, obj=None, name=None, mode='a+', format=None, indices=None, axis=0, verbose=False, indexing=None, **kwargs):
    """ Write DimArray instance to file

    Parameters
    ----------
    f : file name or netCDF file handle
    name : str, optional
        variable name, optional if `name` attribute already defined.
    mode : str, optional
        {write_modes}
        Default mode is 'a+'
    {netCDF4}
    {indexing}

    See Also
    --------
    read_nc
    Dataset.write_nc
    """
    if not name and hasattr(obj, "name"): name = obj.name
    assert name, "invalid variable name !"

    # control wether file name or netCDF handle
    f, close = _check_file(f, mode=mode, verbose=verbose, format=format, **kwargs)

    DimArrayOnDisk(f, name).write(indices=indices, values=obj, axis=axis, indexing=indexing)

    if close:
        f.close()

example_doc_to_recycle = """
    >>> tmpdir = getfixture('tmpdir').strpath # some temporary directory (py.test)
    >>> outfile = os.path.join(tmpdir, 'test.nc')
    >>> da.write_nc(outfile,'myvar', axes=[[1,2,3,4],['a','b','c']], dims=['dim1', 'dim2'], mode='w')
    >>> a = DimArray([11, 22, 33, 44], axes=[[1, 2, 3, 4]], dims=('dim1',)) # some array slice 
    >>> da.write_nc(outfile, a, 'myvar', indices='b', axis='dim2') 
    >>> da.write_nc(outfile, [111,222,333,444], 'myvar', indices='a', axis='dim2') 
    >>> da.read_nc(outfile,'myvar')
    dimarray: 8 non-null elements (4 null)
    0 / dim1 (4): 1 to 4
    1 / dim2 (3): a to c
    array([[ 111.,   11.,   nan],
           [ 222.,   22.,   nan],
           [ 333.,   33.,   nan],
           [ 444.,   44.,   nan]])
    """

##
## write to file
##
@format_doc(netCDF4=_doc_write_nc, indexing=_doc_indexing_write, write_modes=_doc_write_modes)
def write_nc(f, obj=None, *args, **kwargs):
    """  Write DimArray or Dataset to file or Create Empty netCDF file

    This function is a wrapper whose accepted parameters vary depending
    on the object to write.

    Parameters
    ----------
    f : file name or netCDF file handle
    obj : DimArray or Dataset or Axes instance
        An Axes instance will create an empty variable.
    name : str, optional, ONLY IF obj is a DimArray  
        variable name, optional if `name` attribute is already defined.
    dtype : variable type, optional, ONLY IF obj is an Axes object
    mode : `str`, optional
        {write_modes}
        default is 'a+'
    {indexing}
    {netCDF4}

    See Also
    --------
    read_nc, DimArray.write_nc, Dataset.write_nc
    """
    if isinstance(obj, Dataset):
        _write_dataset(f, obj, *args, **kwargs)

    elif isinstance(obj, DimArray) or isinstance(obj, np.ndarray) or isinstance(obj, list):
        _write_variable(f, obj, *args, **kwargs)

    else:
        raise TypeError("only DimArray or Dataset types allowed, \
                or provide variable name (first argument) and axes parameters to create empty variable.\
                \nGot first argument {}:{}".format(type(obj), obj))

#
# display summary information
#

def summary_nc(fname, name=None, metadata=False):
    """ Print summary information about the content of a netCDF file

    Parameters
    ----------
    fname : netCDF file name
    name : variable name, optional
    metadata : bool, optional (default to False)
        if True, also display metadata

    Returns
    -------
    None (print message to screen)

    See Also
    --------
    dimarray.DatasetOnDisk
    """
    with DatasetOnDisk(fname) as obj:
        if name is not None:
            obj = obj[name]
        print(obj.__repr__(metadata=metadata))


@format_doc(write_modes=_doc_write_modes)
def _check_file(f, mode='r', verbose=False, format=None):
    """ open a netCDF4 file

    Parameters
    ----------
    f : file name (str) or netCDF file handle
    mode: changed from original 'r','w','r' & clobber option:

    mode : `str`
        read or write modes

        - 'r': read mode
        {write_modes}

    format: passed to netCDF4.Dataset, only relevatn when mode = 'w', 'w-', 'a+'
        'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT'
         
    Returns
    -------
    f : netCDF file handle
    close: `bool`, `True` if input f indicated file name
    """
    close = False
    clobber = False
    format = format or get_option('io.nc.format')

    if mode == 'w-':
        mode = 'w'

    elif mode == 'w':
        clobber = True

    # mode 'a+' appends if file exists, otherwise create new variable
    elif mode == 'a+' and not isinstance(f, nc.Dataset):
        if os.path.exists(f): mode = 'a'
        else: mode = 'w'

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

#
# Append documentation to dimarray's methods
#
DimArray.write_nc.__func__.__doc__ = _write_variable.__doc__
Dataset.write_nc.__func__.__doc__ = _write_dataset.__doc__

DimArray.read_nc.__func__.__doc__ = _read_variable.__doc__
Dataset.read_nc.__func__.__doc__ = _read_dataset.__doc__
