""" NetCDF I/O to access and save geospatial data
"""
import os
import netCDF4 as nc
import numpy as np
#from geo.data.index import to_slice, _slice3D

from dimarray.dataset import Dataset
from dimarray.core import DimArray, Axis, Axes

__all__ = ['read',"summary"]

#
# wrapper functions which will make it to the api
#

def summary(fname):
    """ print info about netCDF file
    
    input: 
	fname: netCDF file name
    """
    print fname+":\n"+"-"*len(fname)
    nms, _ = scan(fname, verbose=False)
    header = "Dataset of %s variables" % (len(nms))
    if len(nms) == 1: header = header.replace('variables','variable')
    print header
    print read_dimensions(fname, verbose=False)
    for nm in nms:
	dims, shape = scan_var(fname, nm, verbose=False)
	print nm,":", ", ".join(dims)

    #attr = read_attributes(f, name):

def read(f, nms=None, *args, **kwargs):
    """  Read one or several variables from a netCDF file

    Input:
	f	: file name or buffer
	nms	: variable name(s) to read: list or str

    >>> data = read('test.nc')  # load full file
    >>> data = read('test.nc','dynsealevel') # only one variable
    >>> data = read('test.nc','dynsealevel', {"time":slice(2000,2100), "lon":slice(50,70), "lat":42})  # load only a chunck of the data
    """
    if nms is not None and isinstance(nms, str):
	obj = read_variable(f, nms, *args, **kwargs)
    else:
	obj = read_dataset(f, nms, *args, **kwargs)

    return obj


#def write(f, name, data, axes=None, dims=None, mode='w', **metadata):
#    """ Initialize a DimArray and write to netCDF
#
#    Input:
#	f	: file name or buffer
#	name	: variable name
#	data	: numpy array
#	axes	: list of numpy arrays (or Axes objects)
#	dims	: dims of the axes (the dimensions) 
#	mode    : mode to access the netcdf: 'w' (default, overwrite) or 'a' (append)
#	**metadata: for convenience, can pass dimensions as keyword arguments (beware the order is lost)
#
#    data, axes, dims, **metadata are passed to dimarray to instantiate a DimArray, see corresponding doc for details.
#
#    Provide dimensions as a list:
#
#    >>> write("test.nc", "testvar", np.randn(20, 30), [np.linspace(-90,90,20), np.linspace(-180,180,30)],['lat','lon'])
#    """
#    #axes = Axes.from_list(axes, dims)
#    obj = DimArray.from_list(data, axes, dims, **metadata) # initialize a dimarray
#    obj.write(f, name, mode=mode) # write to file


#
# read from file
# 


def read_dimensions(f, name=None, ix=slice(None), verbose=False):
    """ return an Axes object

    name, optional: variable name

    return: Axes object
    """
    f, close = check_file(f, mode='r', verbose=verbose)

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

	values = f.variables[k][ix]

	# replace unicode by str as a temporary bug fix (any string seems otherwise to be treated as unicode in netCDF4)
	if values.size > 0 and type(values[0]) is unicode:
	    for i, val in enumerate(values):
		if type(val) is unicode:
		    values[i] = str(val)

	axes.append(Axis(values, k))

    if close: f.close()

    return axes

def read_attributes(f, name=None, verbose=False):
    """ Read netCDF attributes
    """
    f, close = check_file(f, mode='r', verbose=verbose)
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

def read_variable(f, v, indices=None, axis=0, *args, **kwargs):
    """ read one variable from netCDF4 file
    Input:

	f    	    : file name or file handle
	v         : netCDF variable name to extract
	indices, axis, *args, **kwargs: passed to take

	Please see help on `Dimarray.take` for more information.

    Returns:

	DimArray object 

    >>> data = read_variable('myfile.nc','dynsealevel')  # load full file
    >>> data = read_variable('myfile.nc','dynsealevel', {"time":2100, "lon":slice(70,None)})  # load only a chunck of the data
    """
    kw = _extract_kw(kwargs, ('verbose',))
    f, close = check_file(f, mode='r', **kw)

    # Construct the indices
    axes = read_dimensions(f, v)

    if indices is None:
	newaxes = axes
	newdata = f.variables[v][:]

	# scalar variables come out as arrays ! Fix that.
	if len(axes) == 0:
	    assert np.size(newdata) == 1, "inconsistency betwwen axes and data"
	    assert np.ndim(newdata) == 1, "netCDF seems to have fixed that bug, just remove this line !"
	    newdata = newdata[0]

    else:

	try:
	    ix = axes.loc(indices, axis=axis, *args, **kwargs)
	except IndexError, msg:
	    raise
	    raise IndexError(msg)

	# slice the data and dimensions
	newaxes_raw = [ax[ix[i]] for i, ax in enumerate(axes)] # also get the appropriate axes
	newaxes = [ax for ax in newaxes_raw if isinstance(ax, Axis)] # remove singleton values
	newdata = f.variables[v][ix]

    # initialize a dimarray
    obj = DimArray(newdata, newaxes)
    obj.name = v

    # Read attributes
    attr = read_attributes(f, v)
    for k in attr:
	setattr(obj, k, attr[k])

    # close netCDF if file was given as file name
    if close:
	f.close()

    return obj

#read.__doc__ += read_variable.__doc__

def read_dataset(f, nms=None, *args, **kwargs):
    """ read several (or all) names from a netCDF file

    nms : list of variables to read (default None for all variables)
    """
    kw = _extract_kw(kwargs, ('verbose',))
    f, close = check_file(f, 'r', **kw)

    # automatically read all variables to load (except for the dimensions)
    if nms is None:
	nms, dims = scan(f)

#    if nms is str:
#	nms = [nms]

    gen = dict()
    for nm in nms:
	gen[nm] = read_variable(f, nm, *args, **kwargs)
    gen = Dataset(gen, keys=nms)

    if close: f.close()

    return gen


#
# write to file
#
def write_obj(f, obj, *args, **kwargs):
    """  call write_dataset or write_variable
    """
    import core
    if isinstance(obj, core.DimArray):
	write_variable(f, obj, *args, **kwargs)

    else:
	write_dataset(f, obj, *args, **kwargs)

def write_dataset(f, obj, mode='w-'):
    """ write object to file
    """
    f, close = check_file(f, mode)
    nms = obj.keys()
    for nm in obj:
	write_variable(f, obj[nm], nm)

    if close: f.close()

def write_variable(f, obj, name=None, mode='w-', **verb):
    """ save DimArray instance to file

    f	: file name or netCDF file handle
    obj : DimArray object
    name: variable name
    mode: 'w' or 'a' 
    """
    if not name and hasattr(obj, "name"): name = obj.name
    assert name, "invalid variable name !"

    # control wether file name or netCDF handle
    f, close = check_file(f, mode=mode, **verb)

    # Create dimensions
    for dim in obj.dims:

	if not dim in f.dimensions:
	    val = obj.axes[dim].values
	    f.createDimension(dim, len(val))

	    # check type: so far object-type arrays are strings
	    dtype = val.dtype if val.dtype is not np.dtype('O') else str  
	    v = f.createVariable(dim, dtype, dim)
	    v[:] = val

    # Create Variable
    if name not in f.variables:
	v = f.createVariable(name, obj.dtype, obj.dims)
    else:
	v = f.variables[name]
	#assert tuple(unicode(d) for d in obj.dims) == tuple(v.dimensions), "Dimension name do not correspond {}: {} attempted to write: {}".format(name, tuple(v.dimensions), obj.dims)
	assert obj.dims == tuple(v.dimensions), "Dimension name do not correspond {}: {} attempted to write: {}".format(name, tuple(v.dimensions), obj.dims)

    # Write Variable
    v[:] = obj.values

    # add attributes if any
    for k in obj._metadata.keys():
	if k == "name": continue # 
	try:
	    f.variables[name].setncattr(k, getattr(obj, k))

	except TypeError, msg:
	    raise Warning(msg)

    if close:
	f.close()

#
# util
#

def scan(f, **verb):
    """ get variable names in a netCDF file
    """
    f, close = check_file(f, 'r', **verb)
    nms = f.variables.keys()
    dims = f.dimensions.keys()
    if close: f.close()
    return [n for n in nms if n not in dims], dims

def scan_var(f, v, **verb):
    f, close = check_file(f, 'r', **verb)
    var = f.variables[v]
    dims = var.dimensions
    shape = var.shape
    if close: f.close()
    return dims, shape


def check_file(f, mode='r', verbose=True):
    """ open a netCDF4 file
    """
    close = False

    clobber = False

    if mode == 'w-':
	mode = 'w'

    elif mode == 'w':
	clobber = True

    if not isinstance(f, nc.Dataset):
	fname = f

	# make sure the file does not exist if mode == "w"
	if os.path.exists(fname) and clobber and mode == "w":
	    os.remove(fname)

	try:
	    f = nc.Dataset(fname, mode, clobber=clobber)

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
