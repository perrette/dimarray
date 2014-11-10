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

from dimarray.decorators import format_doc
from dimarray.dataset import Dataset, concatenate_ds, stack_ds
from dimarray.core import DimArray, Axis, Axes
from dimarray.config import get_option
from dimarray.core.metadata import _repr_metadata

__all__ = ['read_nc','summary_nc', 'write_nc', 'read_dimensions']


#
# Global variables 
#
FORMAT='NETCDF4'  # default format to write to netCDF


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

    dimensions_mapping : dict, optional
        mapping between netCDF dimensions and variables in the file
        Keys are dimensions names, values are corresponding variable names.
        if not provided, look for variables with same name as dimension.

    copy_grid_mapping : bool, optional
        if True, any "grid_mapping" attribute pointing to another variable 
        present in the dataset will be replaced by that variable's metadata
        as a dictionary. This can ease transformations.
        Default is False for a variable, True for a Dataset.

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
    tsl: ('time', 'scenario')
    temp: ('time', 'scenario')
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

#
# read from file
# 

def read_dimensions(f, name=None, ix=slice(None), dimensions_mapping=None, verbose=False):
    """ return an Axes object

    Parameters
    ----------
    name : str, optional
        variable name
    ix : integer (position) index
    dimensions_mapping : dict, optional
        mapping between dimension and variable names

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
    for dim in dims:
        if dimensions_mapping is not None:
            # user-provided mapping between dimensions and variable name
            dim_name = dimensions_mapping[dim]
        else:
            dim_name = dim
        axis = _read_dimension(f, dim_name, ix=ix)

        if isinstance(axis, Axis): # we may have a scalar
            axes.append(axis) 

    if close: f.close()

    return axes

def _read_dimension(f, name, ix=None, values_only=False):
    """ Read one netCDF4 dimension as dimarray.Axis

    Parameters
    ----------
    f : netCDF4.Dataset handle
    name : dimension name
    ix : numpy index, optional
    values_only : bool, optional (default False)
        only return array value, instead of creating an Axis object
        (avoid loading metadata, and some Axis checks)

    Returns
    -------
    Axis instance
    """
    if ix is None:
        ix = slice(None)
    if name in f.variables.keys():
        # assume that the variable and dimension have the same name
        values = f.variables[name][ix]
    else:
        # default, dummy dimension axis
        msg = "'{}' dimension not found, define integer range".format(name)
        warnings.warn(msg)
        values = np.arange(len(f.dimensions[name]))[ix]

    # # replace unicode by str as a temporary bug fix (any string seems otherwise to be treated as unicode in netCDF4)
    # if values.size > 0 and type(values[0]) is unicode:
    #     for i, val in enumerate(values):
    #         if type(val) is unicode:
    #             values[i] = str(val)

    # do not produce an Axis object
    if values_only or np.isscalar(values):
        return values

    axis = Axis(values, name)
    
    # add metadata
    if name in f.variables.keys():
        meta = _read_attributes(f, name)
        axis._metadata(meta)

    return axis

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

@format_doc(indexing=_doc_indexing)
def _read_variable(f, name, indices=None, axis=0, indexing='label', verbose=False, dimensions_mapping=None, copy_grid_mapping=False, **kwargs):
    """ Read one variable from netCDF4 file 

    Parameters
    ----------
    f  : file name or file handle
    name : netCDF variable name to extract
    {indexing}
    dimensions_mapping : dict, optional
        mapping between netCDF dimensions and variables in the file
        Keys are dimensions names, values are corresponding variable names.
        if not provided, look for variables with same name as dimension.
    copy_grid_mapping : bool, optional
        if True, any "grid_mapping" attribute pointing to another variable 
        present in the dataset will be replaced by that variable's metadata
        as a dictionary. This can ease transformations.
        Default is True

    Returns
    -------
    Returns a Dimarray instance

    See Also
    --------
    See preferred function `dimarray.read_nc` for more complete documentation 
    """
    f, close = _check_file(f, mode='r', verbose=verbose)

    v = VariableOnDisk(f, name, indexing)
    if indices is None: 
        indices = slice(None)
    obj = v[indices]

    # copy grid_mapping?
    if copy_grid_mapping and hasattr(obj, 'grid_mapping'):
        gm = obj.grid_mapping
        try:
            mapping = _read_variable(f, gm, copy_grid_mapping=False)
            obj.grid_mapping = mapping._metadata()

        except Exception as error:
            warnings.warn(error.message +"\n ==> could not read grid mapping")

    # close netCDF if file was given as file name
    if close:
        f.close()

    return obj

#read_nc.__doc__ += _read_variable.__doc__

def _read_dataset(f, nms=None, dimensions_mapping=None, **kwargs):
    """ Read several (or all) variable from a netCDF file

    Parameters
    ----------
    f : file name or (netcdf) file handle
    nms : list of variables to read (default None for all variables)
    dimensions_mapping : dict, optional
        mapping between netCDF dimensions and variables in the file
        Keys are dimensions names, values are corresponding variable names.
        if not provided, look for variables with same name as dimension.
    **kwargs

    Returns
    -------
    Dataset instance

    See preferred function `dimarray.read_nc` for more complete documentation 
    """
    kw = _extract_kw(kwargs, ('verbose',))
    f, close = _check_file(f, 'r', **kw)

    # when reading a dataset keep grid_mapping as a string
    if 'copy_grid_mapping' not in kwargs:
        kwargs['copy_grid_mapping'] = False

    # automatically read all variables to load (except for the dimensions)
    if nms is None:
        nms, dims = _scan(f)

        if dimensions_mapping is not None:
            nms = [nm for nm in nms if nm not in dimensions_mapping.values()]

#    if nms is str:
#        nms = [nms]

    data = odict()
    for nm in nms:
        data[nm] = _read_variable(f, nm, dimensions_mapping=dimensions_mapping, **kwargs)

    data = Dataset(data)

    # get dataset's metadata
    for k in f.ncattrs():
        setattr(data, k, f.getncattr(k))

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
def _write_dataset(f, obj, mode='w-', indices=None, axis=0, format=FORMAT, verbose=False, **kwargs):
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
    meta = obj._metadata()
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
def _write_variable(f, obj=None, name=None, mode='a+', format=FORMAT, indices=None, axis=0, verbose=False, share_grid_mapping=False, **kwargs):
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
    share_grid_mapping : bool, optional
        if True, write any grid mapping attribute as a 
        separate variable in the dataset, accordingly to CF-conventions
        in order to share that information across several variables.
        Default is False.

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
        assert isinstance(obj, DimArray), "expected a DimArray instance, got {}".format(type(obj))
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
    try:
        a = np.asarray(obj)
        if a.dtype > np.dtype('S1'):
            a = np.asarray(a, dtype=object)
        v[ix] = a
    except:
        print np.asarray(obj)
        print ix
        print np.asarray(obj).dtype
        print v.dtype
        raise

    # add metadata if any
    if not isinstance(obj, DimArray):
        if close:
            f.close()
        return

    meta = obj._metadata()
    for k in meta.keys():
        if k == "name": continue # 
        if meta[k] is None: 
            # do not write empty attribute
            # but delete any existing attribute 
            # of the same name
            if k in f.variables[name].ncattrs:
                f.variables[name].delncattr(k) 
            continue

        # write grid_mapping in a separate variable
        if k == 'grid_mapping' and share_grid_mapping \
                and isinstance(meta[k], dict) \
                and 'grid_mapping_name' in meta[k]:

            # first search whether the mapping is already present
            found = False
            for nm in f.variables.keys():
                ncvar = f.variables[nm]
                if ncvar.size == 1 and hasattr(ncvar, 'grid_mapping_name'):
                    test_grid_mapping = {kk: getattr(ncvar, kk) for kk in ncvar.ncattrs()}
                    if test_grid_mapping == meta['grid_mapping']:
                        found = True
                        break
            if found: 
                meta[k] = nm # point toward the new name
                    
            else:
                name0 = "mapping"
                name = name0
                i = 0
                while name in f.variables.keys():
                    i += 1
                    name = name0+str(i)
                    assert i < 100, 'infinite look'

                try:
                    mapping = f.createVariable(name, np.dtype('S1'), ())
                    for kk, val in meta['grid_mapping'].iteritems():
                        mapping.setncattr(kk, val)
                    meta[k] = name # point toward the new name
                except TypeError as error:
                    msg = error.message
                    msg += "\n=>could not create grid mapping variable"
                    #warnings.warn(msg)
                    raise Warning(msg)

        try:
            f.variables[name].setncattr(k, meta[k])

        except TypeError, msg:
            raise Warning(msg)

    if close:
        f.close()

@format_doc(netCDF4=_doc_write_nc, write_modes=_doc_write_modes)
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
    f, close = _check_file(f, mode=mode, verbose=verbose, format=format)

    # make sure axes is an Axes instance
    if not isinstance(axes, Axes):
        axes = Axes._init(axes, dims=dims)

    _check_dimensions(f, axes)

    # Create Variable
    nctype = _convert_to_nctype(dtype)
    v = f.createVariable(name, nctype, [ax.name for ax in axes], **kwargs)

    if close: 
        f.close()

    return v

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
    share_grid_mapping : bool, optional
        if True, replace any dict-type grid mapping attribute 
        by a string alias, and write the original dict as a separate variable 
        in the dataset. This is accordingly to CF-conventions, in order to 
        share that information across several variables. Default is False.
    {netCDF4}

    See Also
    --------
    read_nc, DimArray.write_nc, Dataset.write_nc
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


def _convert_to_nctype(dtype):
    """ strings are given "object" type in Axis object
    ==> assume all objects are actually strings
    NOTE: this will fail for other object-typed axes such as tuples
    """
    # if dtype is np.dtype('O'):
    if dtype >= np.dtype('S1'): # all strings, this include objects dtype('O')
        nctype = str 
    else:
        nctype = dtype 
    return nctype

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
            nctype = _convert_to_nctype(ax.dtype)
            v = f.createVariable(dim, nctype, dim)
            v[:] = ax.values

        # add metadata if any
        meta = ax._metadata()
        for k in meta.keys():
            if k == "name": continue # 
            if meta[k] is None: 
                # do not write empty attribute
                # but delete any existing attribute 
                # of the same name
                if k in f.variables[name].ncattrs:
                    f.variables[name].delncattr(k) 
                continue
            try:
                f.variables[dim].setncattr(k, meta[k])

            except TypeError, msg:
                raise Warning(msg)

    if close: f.close()

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
    dimarray.io.nc._summary_repr : get the associated string
    """
    print(_summary_repr(fname, name))

def _summary_repr(fname, name=None, metadata=False):
    """ print info about netCDF dataset or one variable
    """
    # open file for reading
    f, close = _check_file(fname, 'r')

    if name is None:
        str_ = _summary_repr_dataset(f, metadata=metadata)
    else:
        str_ = _summary_repr_variable(f, name, metadata=metadata)

    # close file
    if close: f.close()

    return str_

def _summary_repr_dataset(f, metadata=False):
    """ print info about netCDF file
    """
    # variable names
    nms = [nm for nm in f.variables.keys() if nm not in f.dimensions.keys()]

    # header
    header = "Dataset of %s variables (on disk)" % (len(nms))
    if len(nms) == 1: header = header.replace('variables','variable')

    lines = []
    lines.append(header)
    
    # display dimensions name, size, first and last value
    lines.append(_summary_repr_dimensions(f, f.dimensions.keys()))

    # display variables name, shape and dimensions
    for nm in nms:
        dims = f.variables[nm].dimensions
        line = "{name}: {dims}".format(dims=dims, name=nm)
        lines.append(line)

    # Metadata
    if metadata:
        _summary_repr_append_metadata(lines, f)

    return "\n".join(lines)

def _summary_repr_dimensions(f, dims):
    """ string representation of a list of dimensions
    """
    lines = []
    for i, dim in enumerate(dims):
        size = len(f.dimensions[dim]) # dimension size
        try:
            first, last = f.variables[dim][0], f.variables[dim][size-1]
        except KeyError: # if no variable is found
            first, last = 0, size-1
        line = "{i} / {dim} ({size}): {first} to {last}".format(**locals())
        lines.append(line)

    return "\n".join(lines)

def _summary_repr_variable(f, name, metadata=False):
    var = f.variables[name]
    dims = var.dimensions
    lines = []
    lines.append("NetCDF Variable (on disk): {}".format(name))

    # display dimensions name, size, first and last value
    lines.append(_summary_repr_dimensions(f, dims))

    # Metadata
    if metadata:
        _summary_repr_append_metadata(lines, f, name)

    # var = self.variables[name]
    # line = "array"+ str(var.shape) if var.ndim > 0 else str(var[0])
    line = "array(...)" if var.ndim > 0 else str(var[0])
    lines.append(line)

    return "\n".join(lines)

def _summary_repr_append_metadata(lines, f, name=None):
    " append metadata info to a list, if not empty "
    meta = _read_attributes(f, name)
    if len(meta) > 0:
        lines.append("metadata:")
        lines.append(_repr_metadata(meta))

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

@format_doc(write_modes=_doc_write_modes)
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

#
# Append documentation to dimarray's methods
#
DimArray.write_nc.__func__.__doc__ = _write_variable.__doc__
Dataset.write_nc.__func__.__doc__ = _write_dataset.__doc__

DimArray.read_nc.__func__.__doc__ = _read_variable.__doc__
Dataset.read_nc.__func__.__doc__ = _read_dataset.__doc__


#
# EXPERIMENTAL, ONLY PARTLY CONNECTED TO THE ABOVE
#
# Define an open_nc function return an on-disk Dataset object, more similar to 
# netCDF4's Dataset.
#
class NetCDFOnDisk(object):

    _attributes = ['ds', 'name', '_indexing'] # attributes that are set in the __init__
    # needs to define those for __setattr__ and __getattr__, unless we use an ABC abstract class?

    @property
    def _obj(self):
        pass

    # access netCDF attributes
    def getncattr(self, name):
        return self._obj.getncattr(name)
    def setncattr(self, name, value):
        self._obj.setncattr(name, value)
    def delncattr(self, name):
        self._obj.delncattr(name)
    def ncattrs(self):
        return self._obj.ncattrs()

    def _metadata(self, meta=None):
        if meta is None:
            # return _read_attributes(self.ds, name)
            return {k:self.getncattr(k) for k in self.ncattrs()}
        else:
            for k in meta:
                self.setncattr(k, meta[k])

    # general python class attribute access methods, which are overloaded
    # to return / write axis values
    def __delattr__(self, name):
        if name in self.ncattrs():
            self.delncattr(name)
        elif name in self._attributes:
            raise ValueError("cannot delete attribute: "+str(name))
        else:
            object.__delattr__(self, name) # to have a typical error message

    def __getattr__(self, name):
        """ access axis values via '.' attribute syntax, or metadata
        """
        if name not in self._attributes and name in self.dims:
            val = self._get_label(name)
        elif name in self.ncattrs():
            val = self.getncattr(name)
        else:
            val = self.__getattribute__(name) # this 
        return val

    def __setattr__(self, name, value):
        if name not in self._attributes and name in self.dims:
            self.ds.variables[name][:] = value # the axis class will handle types 
        elif name in self._attributes:
            object.__setattr__(self, name, value) # e.g. 
        else:
            # metadata otherwise
            self.setncattr(name, value)

    @property
    def dims(self):
        raise NotImplementedError('need to be overriden !')

    @property
    def axes(self):
        axes = []
        for dim in self.dims:
            axes.append(_read_dimension(self.ds, dim))
        return Axes(axes) # convert to Axes, for pretty printing

    @property
    def labels(self):
        return tuple([self._get_label(dim) for dim in self.dims])

    def _get_label(self, dim, ix=None):
        " return dimension values "
        return _read_dimension(self.ds, dim, ix=ix, values_only=True)

    def close(self):
        return self.ds.close()

    def __exit__(self, exception_type, exception_value, tracebook):
        self.close()

    def __enter__(self):
        return self

class DatasetOnDisk(NetCDFOnDisk):
    def __init__(self, *args, **kwargs):
        self._indexing = kwargs.pop('_indexing', get_option('indexing.by'))
        self.ds = nc.Dataset(*args, **kwargs)

    def read(self, *args, **kwargs):
        """ Equivalent to dimarray.read_nc
        """
        return _read_dataset(self.ds, *args, **kwargs)

    def keys(self):
        """ all variables except for dimensions
        """
        return [nm for nm in self.ds.variables.keys() if nm not in self.dims]

    @property
    def dims(self):
        return tuple(self.ds.dimensions.keys())

    @property
    def _obj(self):
        return self.ds

    def __repr__(self):
        return _summary_repr_dataset(self.ds)

    def __getitem__(self, name):
        return VariableOnDisk(self.ds, name, _indexing=self._indexing)

    def __setitem__(self, name, value):
        _write_variable(self.ds, value, name)

class VariableOnDisk(NetCDFOnDisk):
    _constructor = DimArray
    def __init__(self, ds, name, _indexing=None):
        if _indexing is None:
            _indexing = get_option('indexing.by')
        self._indexing = _indexing
        self.ds = ds
        self.name = name
    @property
    def _obj(self):
        return self.ds.variables[self.name]
    @property
    def shape(self):
        return self._obj.shape
    @property
    def ndim(self):
        return self._obj.ndim
    @property
    def size(self):
        return self._obj.size
    @property
    def dims(self):
        return tuple(self._obj.dimensions)

    @property
    def values(self):
        return self._obj # simply the variable to be indexed and returns values like netCDF4

    def read(self, *args, **kwargs):
        """ Equivalent to dimarray.read_nc
        """
        return _read_variable(self.ds, self.name, *args, **kwargs)
    
    def __getitem__(self, idx):
        # return self._read_variable(self.ds, self.name, indices=idx, indexing=self._indexing)

        # should always be a tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        # load each dimension as necessary
        indices = ()
        axes = []
        for i, dim in enumerate(self.dims):
            if i >= len(idx):
                ix = slice(None)
            else:
                ix = idx[i]

            # in case of label-based indexing, need to read the whole dimension
            # and look for the appropriate values
            if self._indexing != 'position' and not (type(ix) is slice and ix == slice(None)):
                # find the index corresponding to the required axis value
                lix = ix
                ax = _read_dimension(self.ds, dim)
                ix = ax.loc[ix] # label to position index
                ax = ax[ix] # slice / index axis
            else:
                # position index
                ax = _read_dimension(self.ds, dim, ix=ix)
            # if ix is a scalar, the axis is reduced to a scalar as well
            if isinstance(ax, Axis):
                axes.append(ax)
            indices += (ix,)

        values = self.ds.variables[self.name][indices]

        # scalar variables come out as arrays. 
        if len(axes) == 0 and np.ndim(values) != 0:
            # warnings.warn("netCDF4: scalar variables come out as arrays ! Fix that.")
            assert np.size(values) == 1, "inconsistency betwwen axes and data"
            assert np.ndim(values) == 1
            values = values[0]

        dima = self._constructor(values, axes=axes)

        # add attribute
        dima._metadata(self._metadata())

        # attach metadata
        return dima
        # return self.read(indexing=self._indexing)


    # after xray: add sel, isel, loc, iloc methods
    def isel(self, **indices):
        """ Integer selection by keyword

        Examples
        --------
        >>> 
        """
        # replace int dimensions with str dimensions
        for k in indices:
            if type(k) is int:
                indices[k] = self.dims[k]
        # build in
        indices = tuple([indices[d] if d in indices else slice(None) for d in self.dims])
        return self[indices]

    @property
    def ix(self):
        " toggle between position-based and label-based indexing "
        newindexing = 'label' if self._indexing=='position' else 'position'
        return self.__class__(self.ds, self.name, _indexing=newindexing)

    def __setitem__(self, idx, value):
        return _write_variable(self.ds, self.name, indices=idx, indexing=self._indexing)

    def __repr__(self):
        return _summary_repr_variable(self.ds, self.name)

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
    Dataset of 6 variables (on disk)
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
    dimarray: 610 non-null elements (0 null)
    0 / y1 (10): -3400000.0 to -3175000.0
    1 / x1 (61): -800000.0 to 700000.0
    array(...)

    >>> ds['surfvelmag'].sel(x1=700000, y1=-3400000)
    >>> ds['surfvelmag'].isel(x1=-1, y1=0)

    Need to close the Dataset at the end

    >>> ds.close() # close

    Also usable as context manager

    >>> with da.open_nc(ncfile) as ds:
    ...     dataset = ds.read() # load full data set, same as da.read_nc(ncfile)
    >>> dataset
    Dataset of 6 variables
    0 / y1 (113): -3400000.0 to -600000.0
    1 / x1 (61): -800000.0 to 700000.0
    surfvelmag: ('y1', 'x1')
    lat: ('y1', 'x1')
    lon: ('y1', 'x1')
    surfvely: ('y1', 'x1')
    surfvelx: ('y1', 'x1')
    mapping: nan
    """
    return DatasetOnDisk(file_name, *args, **kwargs)

# class DimensionsOnDisk(list):
#     def __init__(self, ds, names):
#         self.ds = ds
#         self.name = names
#
# class DimensionOnDisk(NetCDFOnDisk):
#     def __init__(self, ds, name):
#         self.ds = ds
#         self.name = name
#     @property
#     def _obj(self):
#         if self.name in self.ds.variables.keys():
#             return self.ds.variables[self.name]
#         else:
#             return None

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
