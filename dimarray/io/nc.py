""" NetCDF I/O to access and save geospatial data
"""
import os
import glob
from collections import OrderedDict as odict
import warnings
import netCDF4 as nc
import numpy as np
import dimarray as da
#from geo.data.index import to_slice, _slice3D

from dimarray.dataset import Dataset, concatenate_ds, stack_ds
from dimarray.core import DimArray, Axis, Axes

__all__ = ['read_nc','summary_nc', 'write_nc', 'read_dimensions']


#
# some common piece of documentation
#
_doc_indexing = """
    indices : `dict`
        provide indices or slice to extract {nm1:val1}
    indexing : `str`
        'values' (default) or 'position' (integer position) similarly to `take`
    tol : None or float
       floating point tolerance when indexing float-arrays (default to None for exact match) 
""".strip()

_doc_indexing_write = """
    {take} 
""".format(take=_doc_indexing).strip()

FORMAT='NETCDF4'

_doc_write_nc = """ 
    format : `str`
        netCDF file format. Default is '{format}' (only accounted for during file creation)
    zlib : `bool`
        Enable zlib compression if True. Default is False (no compression).
    complevel : `int`
        integer between 1 and 9 describing the level of compression desired. Ignored if zlib=False.
    **kwargs : key-word arguments
        Any additional keyword arguments accepted by `netCDF4.Dataset.CreateVariable`
    
    Notes:
    ------
    See the netCDF4-python module documentation for more information about the use
    of keyword arguments to write_nc.
""".strip().format(format=FORMAT)

_doc_write_modes = """Several write modes are available:

        - 'w' : write, overwrite if file if present (clobber=True)
        - 'w-': create new file, but raise Exception if file is present (clobber=False)
        - 'a' : append, raise Exception if file is not present
        - 'a+': append if file is present, otherwise create"""


def summary_nc(fname):
    print(_summary_repr(fname))

def _summary_repr(fname):
    """ print info about netCDF file
    
    input: 
        fname: netCDF file name
    """
    lines = []
    lines.append( fname+":\n"+"-"*len(fname))
    nms, _ = _scan(fname, verbose=False)
    header = "Dataset of %s variables" % (len(nms))
    if len(nms) == 1: header = header.replace('variables','variable')
    lines.append(header)
    lines.append(repr(read_dimensions(fname, verbose=False)))
    for nm in nms:
        dims, shape = _scan_var(fname, nm, verbose=False)
        line = nm+": "+repr(dims)
        lines.append(line)

    return "\n".join(lines)

    #attr = _read_attributes(f, name):

def read_nc(f, nms=None, *args, **kwargs):
    """  Read one or several variables from one or several netCDF file

    Parameters
    ----------
    f : str or netCDF handle
        netCDF file to read from or regular expression
    nms : None or list or str, optional
        variable name(s) to read
        default is None
    %s

    Returns
    -------
    obj : DimArray or Dataset
        depending on whether a (single) variable name is passed as argument (nms) or not

    Notes
    -----
    indexing parameters similar to `take` are accepted:
    additional keyword arguments depend on whether one or several files, one or several variables are required for reading (see below)

    Several cases:

    a) Single file

       *args, **kwargs are passed to DimArray.take
       It includes: indices=, axis=, indexing=, position=, tol=

       Please see help on `Dimarray.take` for more information.

    b) Several files

       same keywords arguments are allowed except except for:

       - `axis` now has another meaning. It is not used for indexing but to 
       pick the record dimension (in case of concatenate) or the new dimension 
       along which to join the datasets (in case of stack). 
       Note that indexing is still possible via `indices=` (in dictionary form)
       see DimArray.take for more information.

       - `align=True` in stack mode in order to align datasets prior to 
       concatenation (re-index axes). 

       - keys: sequence, optional
            to be passed to stack_ds, if axis is not 
            part of the dataset

        - align: bool, optional
        if True, reindex axis prior to stacking (default to False)

        - concatenate_only: bool, optional
            if True, only concatenate along existing 
        axis (and raise error if axis not existing)

       Please see help on stack_ds and concatenate_ds for more information
       
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
    dimensions: 'time', 'scenario'
    0 / time (451): 1850 to 2300
    1 / scenario (5): historical to rcp85
    tsl: ('time', 'scenario')
    temp: ('time', 'scenario')
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
    >>> temp.reset_axis(getmodel, axis='model', inplace=True) # would return a copy if inplace is not specified
    >>> temp
    dimarray: 9114 non-null elements (6671 null)
    dimensions: 'model', 'time', 'scenario'
    0 / model (7): IPSL-CM5A-LR to CSIRO-Mk3-6-0
    1 / time (451): 1850 to 2300
    2 / scenario (5): historical to rcp85
    array(...)
    
    This works on datasets as well:

    >>> ds = da.read_nc(direc+'/cmip5.*.nc', align=True, axis='model')
    >>> ds.reset_axis(getmodel, axis='model')
    Dataset of 2 variables
    dimensions: 'model', 'time', 'scenario'
    0 / model (7): IPSL-CM5A-LR to CSIRO-Mk3-6-0
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

    # multi-file ?
    if type(f) is list:
        mf = True
    else:
        mf = False

    # Read a single file
    if not mf:

        # single variable ==> DimArray
        if nms is not None and isinstance(nms, str):
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

read_nc.__doc__ = read_nc.__doc__ % _doc_indexing   # format's {} fails because of dictionary syntax in examples {}

#
# read from file
# 

def read_dimensions(f, name=None, ix=slice(None), verbose=False):
    """ return an Axes object

    Parameters
    ----------
    name: `str`, optional
        variable name

    Returns
    -------
    Axes object
    """
    f, close = _check_file(f, mode='r', verbose=verbose)

    # dimensions associated to the variable to load
    if name is None:
        dims = f.dimensions.keys()
    else:
        try:
            dims = f.variables[name].dimensions
        except KeyError:
            print "Available variable names:",f.variables.keys()
            raise

    dims = [str(d) for d in dims] # conversion to string

    # load axes
    axes = Axes()
    for k in dims:

        try:
            values = f.variables[k][ix]
        except KeyError:
            msg = "'{}' dimension not found, define integer range".format(k)
            warnings.warn(msg)
            values = np.arange(len(f.dimensions[k]))

        # replace unicode by str as a temporary bug fix (any string seems otherwise to be treated as unicode in netCDF4)
        if values.size > 0 and type(values[0]) is unicode:
            for i, val in enumerate(values):
                if type(val) is unicode:
                    values[i] = str(val)

        axes.append(Axis(values, k))

    if close: f.close()

    return axes

def _read_attributes(f, name=None, verbose=False):
    """ Read netCDF attributes

    name: variable name
    """
    f, close = _check_file(f, mode='r', verbose=verbose)
    attr = {}
    if name is None:
        var = f
    else:
        var = f.variables[name]
    for k in var.ncattrs():
        if k.startswith('_'): 
            continue # do not read "hidden" attributes
        attr[k] = var.getncattr(k)

    if close: f.close()
    return attr

def _extract_kw(kwargs, argnames, delete=True):
    """ extract check 
    """
    kw = {}
    for k in kwargs.copy():
        if k in argnames:
            kw[k] = kwargs[k]
            if delete: del kwargs[k]
    return kw

def _read_variable(f, name, indices=None, axis=0, verbose=False, **kwargs):
    """ Read one variable from netCDF4 file 

    Parameters
    ----------
    f              : file name or file handle
    name           : netCDF variable name to extract
    indices, axis, **kwargs: similar to `DimArray.take`

    Returns
    -------
    Returns a Dimarray instance

    See Also
    --------
    See preferred function `dimarray.read_nc` for more complete documentation 
    """
    f, close = _check_file(f, mode='r', verbose=verbose)

    # Construct the indices
    axes = read_dimensions(f, name)

    if indices is None:
        newaxes = axes
        newdata = f.variables[name][:]

        # scalar variables come out as arrays ! Fix that.
        if len(axes) == 0:
            assert np.size(newdata) == 1, "inconsistency betwwen axes and data"
            assert np.ndim(newdata) == 1, "netCDF seems to have fixed that bug, just remove this line !"
            newdata = newdata[0]

    else:

        try:
            ix = axes.loc(indices, axis=axis, **kwargs)
        except IndexError, msg:
            raise
            raise IndexError(msg)

        # slice the data and dimensions
        newaxes_raw = [ax[ix[i]] for i, ax in enumerate(axes)] # also get the appropriate axes
        newaxes = [ax for ax in newaxes_raw if isinstance(ax, Axis)] # remove singleton values
        newdata = f.variables[name][ix]

    # initialize a dimarray
    obj = DimArray(newdata, newaxes)
    obj.name = name

    # Read attributes
    attr = _read_attributes(f, name)
    for k in attr:
        setattr(obj, k, attr[k])

    # close netCDF if file was given as file name
    if close:
        f.close()

    return obj

#read_nc.__doc__ += _read_variable.__doc__

def _read_dataset(f, nms=None, **kwargs):
    """ Read several (or all) variable from a netCDF file

    Parameters
    ----------
    f   : file name or (netcdf) file handle
    nms : list of variables to read (default None for all variables)
    **kwargs

    Returns
    -------
    Dataset instance

    See preferred function `dimarray.read_nc` for more complete documentation 
    """
    kw = _extract_kw(kwargs, ('verbose',))
    f, close = _check_file(f, 'r', **kw)

    # automatically read all variables to load (except for the dimensions)
    if nms is None:
        nms, dims = _scan(f)

#    if nms is str:
#        nms = [nms]

    data = dict()
    for nm in nms:
        data[nm] = _read_variable(f, nm, **kwargs)

    data = Dataset(**data)

    # get dataset's metadata
    for k in f.ncattrs():
        setattr(data, k, f.getncattr(k))

    if close: f.close()

    return data

def _read_multinc(fnames, nms=None, axis=None, keys=None, align=False, concatenate_only=False, **kwargs):
    """ read multiple netCDF files 

    Parameters
    ----------
    fnames: list of file names or file handles to be read
    nms: variable names to be read
    axis: string, optional
        dimension along which the files are concatenated 
        (created as new dimension if not already existing)
    keys: sequence, optional
        to be passed to stack_nc, if axis is not part of the dataset
    align: `bool`, optional
        if True, reindex axis prior to stacking (default to False)
    concatenate_only: `bool`, optional
        if True, only concatenate along existing axis (and raise error if axis not existing) 

    **kwargs: keyword arguments passed to io.nc._read_variable  (cannot 
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
	if keys is None: warnings.warn('keys argument is not used')
        ds = concatenate_ds(datasets, axis=axis)

    else:
        # use file name as keys by default
        if keys is None:
            keys = [os.path.splitext(fn)[0] for fn in fnames]
        ds = stack_ds(datasets, axis=axis, keys=keys, align=align)

    return ds

##
## write to file
##
def write_nc(f, obj=None, *args, **kwargs):
    """  Write DimArray or Dataset to file or Create Empty netCDF file

    Parameters
    ----------
    Expected parameters depend on object to write.

    ### Create Empty netCDF file  ###
    {axes}

    ### DimArray ###
    {dimarray}

    ### Dataset ###
    {dataset}
    """
    if isinstance(obj, Dataset):
        _write_dataset(f, obj, *args, **kwargs)

    elif isinstance(obj, str):
        name = obj
        _createVariable(f, name, *args, **kwargs)

    elif isinstance(obj, DimArray) or isinstance(obj, np.ndarray) or isinstance(obj, list):
        _write_variable(f, obj, *args, **kwargs)

    else:
        raise TypeError("only DimArray or Dataset types allowed, \
                or provide variable name (first argument) and axes parameters to create empty variable.\
                \nGot first argument {}:{}".format(type(obj), obj))

#def _write_empty_variable(f, obj, name, **kwargs):
##_createVariable.__doc__ = _createVariable.__doc__.format(netCDF4=_doc_write_nc)

#_doc_write_nc = """ 

#def write_attributes(f, obj):

def _write_dataset(f, obj, mode='w-', indices=None, axis=0, format=FORMAT, verbose=False, **kwargs):
    """ Write Dataset to netCDF file

    Parameters
    ----------
    f : `str` or netCDF handle
        netCDF file to write to

    mode : `str`
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
    meta = obj._metadata
    for k in meta.keys():
        f.setncattr(k, meta[k])

    if close: f.close()



def _write_variable(f, obj=None, name=None, mode='a+', format=FORMAT, indices=None, axis=0, verbose=False, **kwargs):
    """ Write DimArray instance to file

    Parameters
    ----------
    f      : file name or netCDF file handle
    name   : `str`, optional
        variable name, optional if `name` attribute already defined.
    mode: `str`, optional
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
    f, close = _check_file(f, mode=mode, verbose=verbose, format=format)

    # create variable if necessary
    if name not in f.variables:
        assert isinstance(obj, DimArray), "a must be a DimArray"
        v = _createVariable(f, name, obj.axes, dtype=obj.dtype, **kwargs)

    else:
        v = f.variables[name]

        # remove dimension check to write slices
        #if isinstance(obj, DimArray) and obj.dims != tuple(v.dimensions):
        #    raise ValueError("Dimension name do not correspond {}: {} attempted to write: {}".format(name, tuple(v.dimensions), obj.dims))

    # determine indices
    if indices is None:
        ix = slice(None)

    else:
        axes = read_dimensions(f, name)
        try:
            ix = axes.loc(indices, axis=axis)
        except IndexError, msg:
            raise
            raise IndexError(msg)

    # Write Variable
    v[ix] = np.asarray(obj)

    # add metadata if any
    if isinstance(obj, DimArray):
        meta = obj._metadata
        for k in meta.keys():
            if k == "name": continue # 
            try:
                f.variables[name].setncattr(k, meta[k])

            except TypeError, msg:
                raise Warning(msg)

    if close:
        f.close()

def _createVariable(f, name, axes, dims=None, dtype=float, verbose=False, mode='a+', format=FORMAT, **kwargs):
    """ Create empty netCDF4 variable from axes

    Parameters
    ----------
    f: string or netCDF4.Dataset file handle
    name : variable name
    axes : Axes's instance or list of numpy arrays (axis values)
    dims: sequence, optional
        dimension names (to be used in combination with axes to create Axes instance)
    dtype : optional, variable type
    mode : `str`, optional
        {write_modes}
        default is 'a+'

    {netCDF4}

    Examples
    --------
    >>> tmpdir = getfixture('tmpdir').strpath # some temporary directory (py.test)
    >>> outfile = os.path.join(tmpdir, 'test.nc')
    >>> da.write_nc(outfile,'myvar', axes=[[1,2,3,4],['a','b','c']], dims=['dim1', 'dim2'], mode='w')
    >>> a = da.array([11, 22, 33, 44], axes=[[1, 2, 3, 4]], dims=('dim1',)) # some array slice 
    >>> da.write_nc(outfile, a, 'myvar', indices='b', axis='dim2') 
    >>> da.write_nc(outfile, [111,222,333,444], 'myvar', indices='a', axis='dim2') 
    >>> da.read_nc(outfile,'myvar')
    dimarray: 8 non-null elements (4 null)
    dimensions: 'dim1', 'dim2'
    0 / dim1 (4): 1 to 4
    1 / dim2 (3): a to c
    array([[ 111.,   11.,   nan],
           [ 222.,   22.,   nan],
           [ 333.,   33.,   nan],
           [ 444.,   44.,   nan]])
    """
    f, close = _check_file(f, mode=mode, verbose=verbose, format=format)

    # make sure axes is an Axes instance
    if not isinstance(axes, Axes):
        axes = Axes._init(axes, dims=dims)

    _check_dimensions(f, axes)

    # Create Variable
    v = f.createVariable(name, dtype, [ax.name for ax in axes], **kwargs)

    if close: 
        f.close()

    return v


# add doc for keyword arguments
_write_dataset.__doc__ = _write_dataset.__doc__.format(netCDF4=_doc_write_nc, indexing=_doc_indexing_write, write_modes=_doc_write_modes)
_write_variable.__doc__ = _write_variable.__doc__.format(netCDF4=_doc_write_nc, indexing=_doc_indexing_write, write_modes=_doc_write_modes)
_createVariable.__doc__ = _createVariable.__doc__.format(netCDF4=_doc_write_nc, write_modes=_doc_write_modes)

write_nc.__doc__ = write_nc.__doc__.format(dimarray=_write_variable.__doc__, dataset=_write_dataset.__doc__, axes=_createVariable.__doc__)


def _check_dimensions(f, axes, **verb):
    """ create dimensions if not already existing

    Parameters
    ----------
    f : `str` or netCDF handle
        netCDF file to write to
    axes: list of Axis objects
    **verb: passed to _check_file (e.g. verbose=False)
    """
    f, close = _check_file(f, mode='a+', **verb)
    for ax in axes:
        dim = ax.name
        if not dim in f.dimensions:
            f.createDimension(dim, ax.size)

            # strings are given "object" type in Axis object
            # ==> assume all objects are actually strings
            # NOTE: this will fail for other object-typed axes such as tuples
            # any other idea welcome
            if ax.dtype is np.dtype('O'):
                dtype = str 
            else:
                dtype = ax.dtype 

            v = f.createVariable(dim, dtype, dim)
            v[:] = ax.values

        # add metadata if any
        meta = ax._metadata
        for k in meta.keys():
            if k == "name": continue # 
            try:
                f.variables[dim].setncattr(k, meta[k])

            except TypeError, msg:
                raise Warning(msg)

    if close: f.close()



#
# util
#

def _scan(f, **verb):
    """ get variable names in a netCDF file
    """
    f, close = _check_file(f, 'r', **verb)
    nms = f.variables.keys()
    dims = f.dimensions.keys()
    if close: f.close()
    return [n for n in nms if n not in dims], dims

def _scan_var(f, v, **verb):
    f, close = _check_file(f, 'r', **verb)
    var = f.variables[v]
    dims = var.dimensions
    shape = var.shape
    if close: f.close()
    return dims, shape


def _check_file(f, mode='r', verbose=False, format='NETCDF4'):
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

_check_file.__doc__ = _check_file.__doc__.format(write_modes=_doc_write_modes)

#
# Append documentation to dimarray's methods
#
DimArray.write_nc.__func__.__doc__ = _write_variable.__doc__
Dataset.write_nc.__func__.__doc__ = _write_dataset.__doc__

DimArray.read_nc.__func__.__doc__ = _read_variable.__doc__
Dataset.read_nc.__func__.__doc__ = _read_dataset.__doc__

##
## Create a wrapper which behaves similarly to a Dataset and DimArray object
##
## ==> NEED TO FIX BUGS BEFORE USE (netCDF4 crashes)
##
#class NCGeneric(object):
#    """ generic netCDF class dealing with attribute I/O
#    """
#    def _getnc(self):
#        """
#        returns:
#        nc: Dataset or Variable handle
#        f : Dataset handle (same as nc for NCDataset)
#        close: bool, close Dataset after action?
#        """
#        raise NotImplementedError()
#
#    def setncattr(self, nm, val):
#        nc, f, close = self._getnc(mode='w-')
#        nc.setncattr(nm, val)
#        if close: f.close()
#
#    def delncattr(self, nm, val):
#        nc, f, close = self._getnc(mode='w-')
#        nc.delncattr(nm, val)
#        if close: f.close()
#
#    def getncattr(self, nm):
#        nc, f, close = self._getnc(mode='r')
#        attr = nc.getncattr(nm)
#        if close: f.close()
#        return attr
#
#    def ncattrs(self):
#        nc, f, close = self._getnc(mode='r')
#        attr = nc.ncattrs()
#        if close: f.close()
#        return attr
#
#    def __getattr__(self, nm):
#        """ get attribute can also retrieve numpy-like properties
#        """
#        nc, f, close = self._getnc(mode='r')
#        attr = getattr(nc, nm)
#        if close: f.close()
#        return attr
#
#    __delattr__ = delncattr
#    __setattr__ = setncattr
#
#class NCDataset(NCGeneric):
#    """
#    """
#    def __init__(self, f, keepopen=False):
#        """ register the filename
#        """
#        if keepopen:
#            f, close = _check_file(f, mode='w-')
#
#        self.__dict__.update({'f':f, 'keepopen':keepopen}) # by pass setattr
#
#    def read(self, *args, **kwargs):
#        """ Read the netCDF file and convert to Dataset
#        """
#        return read(self.f, *args, **kwargs)
#
#    def write(self, *args, **kwargs):
#        """ Read the netCDF file and convert to Dataset
#        """
#        #f, close = _check_file(self.f, mode='w-', verbose=False)
#        return write_obj(self.f, *args, **kwargs)
#
#    def __getitem__(self, nm):
#        return NCVariable(self.f, nm)
#
#    def __setitem__(self, nm, a):
#        """ Add a variable to netCDF file
#        """
#        return _write_variable(self.f, a, name=nm, verbose=False)
#
##    def __delitem__(self, nm):
##        """ delete a variable
##        
##        NOTE: netCDF4 does not allow deletion of variables because it is not part of netCDF's C api
##        This command use `ncks` from the nco fortran program to copy the dataset on file, 
##        except for the variable to delete
##        """
##        assert isinstance(nm, str), "must be string"
##        if nm not in self.keys():
##            raise ValueError(nm+' not present in dataset')
##        fname = self.f
##        assert isinstance(fname, str), "file name must be string, no netCDF handle"
##        cmd = 'ncks -x -v {nm} {fname} {fname}'.format(nm=nm, fname=fname)
##        print cmd
##        r = os.system(cmd)
##        if r != 0:
##            print r
##            raise Exception('deleting variable failed: you must be on unix with `nco` installed')
#
#    def __repr__(self):
#        """ string representation of the Dataset
#        """
#        try:
#            return _summary_repr(self.f)
#        except IOError:
#            return "empty dataset"
#
#    def _getnc(self, mode, verbose=False):
#        """ used for setting and getting attributes
#        """
#        f, close = _check_file(self.f, mode=mode, verbose=verbose)
#        return f, f, close
#
#    @property
#    def axes(self):
#        return read_dimensions(self.f, verbose=False)
#
#    #
#    # classical ordered dictionary attributes
#    #
#    def keys(self):
#        try:
#            names, dims = _scan(self.f)
#        except IOError:
#            names = []
#
#        return names
#
#    @property
#    def dims(self):
#        try:
#            names, dims = _scan(self.f)
#        except IOError:
#            dims = ()
#
#        return dims
#
#    def len(self):
#        return len(self.keys())
#
#class NCVariable(NCGeneric):
#    """
#    """
#    def __init__(self, f, name, indexing = 'values'):
#        """
#        """
#        # bypass __setattr__, reserved for metadata
#        self.__dict__.update({'f':f,'name':name,'indexing':indexing})
#
#    def __getitem__(self, indices):
#        return self.read(indices, verbose=False)
#
#    def __setitem__(self, indices, values):
#        return self.write(values, indices=indices)
#
#    def _getnc(self, mode, verbose=False):
#        """ get netCDF4 Variable handle 
#        """
#        f, close = _check_file(self.f, mode=mode, verbose=verbose)
#        return f.variables[f.name], f, close
#
#    def read(self, *args, **kwargs):
#        indexing = kwargs.pop('indexing', self.indexing)
#        kwargs['indexing'] = indexing
#        return _read_variable(self.f, self.name, *args, **kwargs)
#
#    def write(self, values, *args, **kwargs):
#        assert 'name' not in kwargs, "'name' is not a valid parameter"
#        indexing = kwargs.pop('indexing', self.indexing)
#        kwargs['indexing'] = indexing
#        return _write_variable(self.f, values, self.name, *args, **kwargs)
#
#    @property
#    def ix(self):
#        return NCVariable(self.f, self.name, indexing='position')
#
#    @property
#    def axes(self):
#        return read_dimensions(self.f, name=self.name)
#
#    def __repr__(self):
#        return "\n".join(["{}: {}".format(self.__class__.__name__, self.name),repr(self.axes)])
