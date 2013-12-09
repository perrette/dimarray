""" NetCDF I/O to access and save geospatial data
"""
import netCDF4 as nc
import numpy as np
#from geo.data.index import to_slice, _slice3D
from collect import Dataset
from axes import Axis, Axes
from lazyapi import dimarray

__all__ = ['read',"summary","write"]

#
# wrapper functions which will make it to the api
#

def summary(fname):
    """ print info about netCDF file
    
    input: 
	fname: netCDF file name
    """
    print read_dimensions(fname)
    nms, _ = scan(fname, verbose=False)
    for nm in nms:
	dims, shape = scan_var(fname, nm, verbose=False)
	print nm,":", " x ".join(dims)

    #attr = read_attributes(f, name):

def read(f, nms=None, *args, **kwargs):
    """  Read one or several variables from a netCDF file

    Input:
	f	: file name or buffer
	nms	: variable name(s) to read: list or str

    >>> data = read('test.nc')  # load full file
    >>> data = read('test.nc','dynsealevel') # only one variable
    >>> data = read('test.nc','dynsealevel', time=(2000,2100), lon=(50,70), lat=42)  # load only a chunck of the data
    """
    if nms is not None and type(nms) is str:
	return read_variable(f, nms, *args, **kwargs)
    else:
	return read_dataset(f, nms, *args, **kwargs)

def write(f, name, data, axes=None, names=None, mode='w', **kwargs):
    """ Initialize a Dimarray and write to netCDF

    Input:
	f	: file name or buffer
	name	: variable name
	data	: numpy array
	axes	: list of numpy arrays (or Axes objects)
	names	: names of the axes (the dimensions) 
	mode    : mode to access the netcdf: 'w' (default, overwrite) or 'a' (append)
	**kwargs: for convenience, can pass dimensions as keyword arguments (beware the order is lost)

    data, axes, names, **kwargs are passed to dimarray to instantiate a Dimarray, see corresponding doc for details.

    Provide dimensions as a list:

    >>> write("test.nc", "testvar", np.randn(20, 30), [np.linspace(-90,90,20), np.linspace(-180,180,30)],['lat','lon'])

    Or as key-word arguments:

    >>> write("test.nc", "testvar", np.randn(20, 30), lat=np.linspace(-90,90,20), lon=np.linspace(-180,180,30))

    Note:   If keyword arguments are given, the order is lost. 
	    In case there is ambiguity in how to order the data 
	    (e.g. two dimensions have the same length), provide arguments as list.
    """
    obj = dimarray(data, axes, names, name=name, **kwargs) # initialize a dimarray
    obj.write(f, mode=mode) # write to file


#
# read from file
# 


def read_dimensions(f, name=None):
    """ return an Axes object

    name, optional: variable name

    return: Axes object
    """
    f, close = check_file(f, mode='r')

    # dimensions associated to the variable to load
    if name is None:
	dims = f.dimensions.keys()
    else:
	dims = f.variables[name].dimensions

    # load axes
    axes = Axes()
    for k in dims:
	axes.append(f.variables[k], k)

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


def read_variable(f, v, ix=None, **kwargs):
    """ read any one variable from netCDF4 file with dimensions "sample","time","lat","lon"
    Input:

	- f    	    : file name or file handle
	- v         : netCDF variable name to extract
	- ix	    : [None] integer index slice for N-D numpy array

	**slices: keyword arguments for dimensions to slice
	- sample    : [None] time
	- time	    : [None] time
	- lat       : [None] latitude
	- lon       : [None] longitude

	- exact     : [False]  # make sure slices match exactly

    Returns:

	Dimarray object (e.g. TimeSeries, Slab, Cube ...)

    >>> data = read_variable('myfile.nc','dynsealevel')  # load full file
    >>> data = read_variable('myfile.nc','dynsealevel', time=2000,2100, lon=50,70)  # load only a chunck of the data

    DOC:

    Any of the dimension have the following types:

    - None: full axis

    - slice: direct slice access like ix above

    - int, float: retrieve corresponding index matching the axis value

    - tuple of int or float: retrieve range matching the axsis value

    For the last two, a nearest neighbour search is performed to retrieve 
    closest location on the axis.  For an exact match, add keyword argument:
    """
    from core import dimarray

    f, close = check_file(f, mode='r')

    axes = read_dimensions(f, v)

    # get the slice object
    if ix is None:
	ix = axes.loc(**kwargs) # get the N-D slice

    # slice the data and dimensions
    newdata = f.variables[v][ix]
    newaxes = [Axis(ax[ix[i]], ax.name) for i, ax in enumerate(axes)] # also get the appropriate axes

    # initialize a dimarray
    obj = dimarray(newdata, newaxes, name=v)

    # Read attributes
    attr = read_attributes(f, v)
    obj.__dict__.update(attr)

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

    gen = Dataset() # 
    for nm in nms:
	gen[nm] = read_variable(f, nm, *args, **kwargs)

    if close: f.close()

    return gen


#
# write to file
#
def write_obj(f, obj, *args, **kwargs):
    """  call write_dataset or write_variable
    """
    import core
    if isinstance(obj, core.Dimarray):
	write_variable(f, obj, *args, **kwargs)

    else:
	write_dataset(f, obj, *args, **kwargs)

def write_dataset(f, obj, mode='w'):
    """ write object to file
    """
    f, close = check_file(f, mode)
    nms = obj.keys()
    for nm in obj:
	write_variable(f, obs[nm], nm)


def write_variable(f, obj, name, mode='w'):
    """ save Dimarray instance to file

    f	: file name or netCDF file handle
    obj : Dimarray object
    name: variable name
    mode: 'w' or 'a' 
    """
    # control wether file name or netCDF handle
    f, close = check_file(f, mode=mode)

    # Create dimensions
    for dim in obj.axes.names:
	if not dim in f.dimensions:
	    val = obj.axes[dim].values
	    f.createDimension(dim, len(val))
	    f.createVariable(dim, val.dtype, dim)
	    f.variables[dim][:] = val

    # Create Variable
    if name not in f.variables:
	f.createVariable(name, obj.dtype, obj.axes.names)

    # Write Variable
    f.variables[name][:] = obj.values

    # add attributes if any
    for k in obj.ncattrs():
	f.variables[name].setncattr(k, getattr(obj, k))

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
	f = nc.Dataset(fname, mode)
	close = True

    return f, close
