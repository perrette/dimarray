""" NetCDF I/O to access and save geospatial data
"""
import os
import netCDF4 as nc
import numpy as np
#from geo.data.index import to_slice, _slice3D

from ..dataset import Dataset
from ..da import DimArray, Axis, Axes

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
    if nms is not None and type(nms) is str:
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
	dims = f.variables[name].dimensions

    # load axes
    axes = Axes()
    for k in dims:
	axes.append(Axis(f.variables[k][ix], k))

    if close: f.close()

    return axes

def read_attributes(f, name=None):
    """ Read netCDF attributes
    """
    f, close = check_file(f, mode='r')
    attr = {}
    if name is None:
	var = f
    else:
	var = f.variables[name]
    for k in var.ncattrs():
	attr[k] = var.getncattr(k)

    if close: f.close()
    return attr


def read_variable(f, v, indices=slice(None), axis=0, *args, **kwargs):
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
    f, close = check_file(f, mode='r')

    # Construct the indices
    axes = read_dimensions(f, v)
    ix = axes.loc(indices, axis=axis, *args, **kwargs)

    # slice the data and dimensions
    newaxes = [ax[ix[i]] for i, ax in enumerate(axes)] # also get the appropriate axes
    newdata = f.variables[v][ix]

    # initialize a dimarray
    obj = DimArray(newdata, newaxes)
    obj.name = v

    # Read attributes
    attr = read_attributes(f, v)
    obj._metadata.update(attr)

    # close netCDF if file was given as file name
    if close:
	f.close()

    return obj

#read.__doc__ += read_variable.__doc__

def read_dataset(f, nms=None, *args, **kwargs):
    """ read several (or all) names from a netCDF file

    nms : list of variables to read (default None for all variables)
    """

    f, close = check_file(f, 'r')

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

def write_dataset(f, obj, mode='a'):
    """ write object to file
    """
    f, close = check_file(f, mode)
    nms = obj.keys()
    for nm in obj:
	write_variable(f, obj[nm], nm)


def write_variable(f, obj, name=None, mode='a'):
    """ save DimArray instance to file

    f	: file name or netCDF file handle
    obj : DimArray object
    name: variable name
    mode: 'w' or 'a' 
    """
    if not name and hasattr(obj, "name"): name = obj.name
    assert name, "invalid variable name !"

    # control wether file name or netCDF handle
    f, close = check_file(f, mode=mode)

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
	assert obj.dims == tuple(v.dimensions), "Dimension name do not correspond {}: {} attempted to write: {}".format(name, tuple(v.dimensions), obj.dims)

    # Write Variable
    v[:] = obj.values

    # add attributes if any
    for k in obj.ncattrs():
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

    if type(f) is str:
	fname = f
	if verbose: 
	    if 'r' in mode:
		print "read from",fname
	    else:
		print "write to",fname

	# make sure the file does not exist if mode == "w"
	if mode=="w" and os.path.exists(fname):
	    os.remove(fname)

	f = nc.Dataset(fname, mode)
	close = True

    return f, close
