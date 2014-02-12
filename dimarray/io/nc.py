""" NetCDF I/O to access and save geospatial data
"""
import os
from collections import OrderedDict as odict
import netCDF4 as nc
import numpy as np
#from geo.data.index import to_slice, _slice3D

from dimarray.dataset import Dataset
from dimarray.core import DimArray, Axis, Axes

__all__ = ['read','summary']

def summary(fname):
    print(summary_repr(fname))

def summary_repr(fname):
    """ print info about netCDF file
    
    input: 
	fname: netCDF file name
    """
    lines = []
    lines.append( fname+":\n"+"-"*len(fname))
    nms, _ = scan(fname, verbose=False)
    header = "Dataset of %s variables" % (len(nms))
    if len(nms) == 1: header = header.replace('variables','variable')
    lines.append(header)
    lines.append(repr(read_dimensions(fname, verbose=False)))
    for nm in nms:
	dims, shape = scan_var(fname, nm, verbose=False)
	line = nm+": "+repr(dims)
	lines.append(line)

    return "\n".join(lines)

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

    name: variable name
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

def read_variable(f, v, indices=None, axis=0, **kwargs):
    """ read one variable from netCDF4 file
    Input:

	f    	    : file name or file handle
	v         : netCDF variable name to extract
	indices, axis, **kwargs: passed to take

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
	    ix = axes.loc(indices, axis=axis, **kwargs)
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

    data = dict()
    for nm in nms:
	data[nm] = read_variable(f, nm, *args, **kwargs)

    data = Dataset(**data)

    # get dataset's metadata
    for k in f.ncattrs():
	setattr(data, k, f.getncattr(k))

    if close: f.close()

    return data


#
# write to file
#
def write_obj(f, obj, *args, **kwargs):
    """  call write_dataset or write_variable
    """
    if isinstance(obj, DimArray):
	write_variable(f, obj, *args, **kwargs)

    elif isinstance(obj, Dataset):
	write_dataset(f, obj, *args, **kwargs)

    else:
	raise TypeError("only DimArray or Dataset types allowed, got {}:{}".format(type(obj), obj))

def write_dataset(f, obj, mode='w-', **kwargs):
    """ write object to file
    """
    f, close = check_file(f, mode, **kwargs)
    nms = obj.keys()
    for nm in obj:
	write_variable(f, obj[nm], nm)

    # set metadata for the whole dataset
    meta = obj._metadata
    for k in meta.keys():
	f.setncattr(k, meta[k])

    if close: f.close()

def check_dimensions(f, axes, **verb):
    """ create dimensions if not already existing

    f : file handle or string
    axes: list of Axis objects
    """
    f, close = check_file(f, mode='a+', **verb)
    for ax in axes:
	dim = ax.name
	if not dim in f.dimensions:
	    f.createDimension(dim, ax.size)

	    # check type: so far object-type arrays are strings
	    dtype = ax.dtype if ax.dtype is not np.dtype('O') else str  
	    v = f.createVariable(dim, dtype, dim)
	    v[:] = ax.values

    if close: f.close()

def createVariable(f, name, axes=None, dtype=float, **kwargs):
    """ create variable from an Axes object

    f: Dataset file handle
    name : variable name
    dtype : variable type
    a: DimArray
    name: variable name if a.name is not defined
    """
    # Create dimensions if necessary
    check_dimensions(f, axes)

    # Create Variable
    return f.createVariable(name, dtype, [ax.name for ax in axes], **kwargs)

#def write_attributes(f, obj):

def write_variable(f, obj, name=None, mode='a+', format='NETCDF4', indices=None, axis=0, verbose=False, **kwargs):
    """ save DimArray instance to file

    f	: file name or netCDF file handle
    obj : DimArray object
    name: variable name
    mode: 'a+' append if possible, otherwise write
    """
    if not name and hasattr(obj, "name"): name = obj.name
    assert name, "invalid variable name !"

    # control wether file name or netCDF handle
    f, close = check_file(f, mode=mode, verbose=verbose, format=format)

    # create variable if necessary
    if name not in f.variables:
	assert isinstance(obj, DimArray), "a must be a DimArray"
	v = createVariable(f, name, obj.axes, dtype=obj.dtype)

    # or just check dimensions
    else:
	v = f.variables[name]

	if isinstance(obj, DimArray) and obj.dims != tuple(v.dimensions):
	    raise ValueError("Dimension name do not correspond {}: {} attempted to write: {}".format(name, tuple(v.dimensions), obj.dims))
    # determine indices
    if indices is None:
	ix = slice(None)

    else:
	axes = read_dimensions(f, name)
	try:
	    ix = axes.loc(indices, axis=axis, **kwargs)
	except IndexError, msg:
	    raise
	    raise IndexError(msg)

    # Write Variable
    v[ix] = np.asarray(obj)

    # add attributes if any
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


def check_file(f, mode='r', verbose=True, format='NETCDF4'):
    """ open a netCDF4 file

    mode: changed from original 'r','w','r' & clobber option:

    'r' : read, raise Exception if file is not present 
    'w' : write, overwrite if file if present (clobber=True)
    'w-': create new file, but raise Exception if file is present (clobber=False)
    'a' : append, raise Exception if file is not present
    'a+': append if file is present, otherwise create

    format: passed to netCDF4.Dataset, only relevatn when mode = 'w', 'w-', 'a+'
	'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT'
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


##
## Create a wrapper which behaves similarly to a Dataset and DimArray object
##
## ==> NEED TO FIX BUGS BEFORE USE (netCDF4 crashes)
##
#class NCGeneric(object):
#    """ generic netCDF class dealing with attribute I/O
#    """
#    def _getnc(self):
#	"""
#	returns:
#	nc: Dataset or Variable handle
#	f : Dataset handle (same as nc for NCDataset)
#	close: bool, close Dataset after action?
#	"""
#	raise NotImplementedError()
#
#    def setncattr(self, nm, val):
#	nc, f, close = self._getnc(mode='w-')
#	nc.setncattr(nm, val)
#	if close: f.close()
#
#    def delncattr(self, nm, val):
#	nc, f, close = self._getnc(mode='w-')
#	nc.delncattr(nm, val)
#	if close: f.close()
#
#    def getncattr(self, nm):
#	nc, f, close = self._getnc(mode='r')
#	attr = nc.getncattr(nm)
#	if close: f.close()
#	return attr
#
#    def ncattrs(self):
#	nc, f, close = self._getnc(mode='r')
#	attr = nc.ncattrs()
#	if close: f.close()
#	return attr
#
#    def __getattr__(self, nm):
#	""" get attribute can also retrieve numpy-like properties
#	"""
#	nc, f, close = self._getnc(mode='r')
#	attr = getattr(nc, nm)
#	if close: f.close()
#	return attr
#
#    __delattr__ = delncattr
#    __setattr__ = setncattr
#
#class NCDataset(NCGeneric):
#    """
#    """
#    def __init__(self, f, keepopen=False):
#	""" register the filename
#	"""
#	if keepopen:
#	    f, close = check_file(f, mode='w-')
#
#	self.__dict__.update({'f':f, 'keepopen':keepopen}) # by pass setattr
#
#    def read(self, *args, **kwargs):
#	""" Read the netCDF file and convert to Dataset
#	"""
#	return read(self.f, *args, **kwargs)
#
#    def write(self, *args, **kwargs):
#	""" Read the netCDF file and convert to Dataset
#	"""
#	#f, close = check_file(self.f, mode='w-', verbose=False)
#	return write_obj(self.f, *args, **kwargs)
#
#    def __getitem__(self, nm):
#	return NCVariable(self.f, nm)
#
#    def __setitem__(self, nm, a):
#	""" Add a variable to netCDF file
#	"""
#	return write_variable(self.f, a, name=nm, verbose=False)
#
##    def __delitem__(self, nm):
##	""" delete a variable
##	
##	NOTE: netCDF4 does not allow deletion of variables because it is not part of netCDF's C api
##	This command use `ncks` from the nco fortran program to copy the dataset on file, 
##	except for the variable to delete
##	"""
##	assert isinstance(nm, str), "must be string"
##	if nm not in self.keys():
##	    raise ValueError(nm+' not present in dataset')
##	fname = self.f
##	assert isinstance(fname, str), "file name must be string, no netCDF handle"
##	cmd = 'ncks -x -v {nm} {fname} {fname}'.format(nm=nm, fname=fname)
##	print cmd
##	r = os.system(cmd)
##	if r != 0:
##	    print r
##	    raise Exception('deleting variable failed: you must be on unix with `nco` installed')
#
#    def __repr__(self):
#	""" string representation of the Dataset
#	"""
#	try:
#	    return summary_repr(self.f)
#	except IOError:
#	    return "empty dataset"
#
#    def _getnc(self, mode, verbose=False):
#	""" used for setting and getting attributes
#	"""
#	f, close = check_file(self.f, mode=mode, verbose=verbose)
#	return f, f, close
#
#    @property
#    def axes(self):
#	return read_dimensions(self.f, verbose=False)
#
#    #
#    # classical ordered dictionary attributes
#    #
#    def keys(self):
#	try:
#	    names, dims = scan(self.f)
#	except IOError:
#	    names = []
#
#	return names
#
#    @property
#    def dims(self):
#	try:
#	    names, dims = scan(self.f)
#	except IOError:
#	    dims = ()
#
#	return dims
#
#    def len(self):
#	return len(self.keys())
#
#class NCVariable(NCGeneric):
#    """
#    """
#    def __init__(self, f, name, indexing = 'values'):
#	"""
#	"""
#	# bypass __setattr__, reserved for metadata
#	self.__dict__.update({'f':f,'name':name,'indexing':indexing})
#
#    def __getitem__(self, indices):
#	return self.read(indices, verbose=False)
#
#    def __setitem__(self, indices, values):
#	return self.write(values, indices=indices)
#
#    def _getnc(self, mode, verbose=False):
#	""" get netCDF4 Variable handle 
#	"""
#	f, close = check_file(self.f, mode=mode, verbose=verbose)
#	return f.variables[f.name], f, close
#
#    def read(self, *args, **kwargs):
#	indexing = kwargs.pop('indexing', self.indexing)
#	kwargs['indexing'] = indexing
#	return read_variable(self.f, self.name, *args, **kwargs)
#
#    def write(self, values, *args, **kwargs):
#	assert 'name' not in kwargs, "'name' is not a valid parameter"
#	indexing = kwargs.pop('indexing', self.indexing)
#	kwargs['indexing'] = indexing
#	return write_variable(self.f, values, self.name, *args, **kwargs)
#
#    @property
#    def ix(self):
#	return NCVariable(self.f, self.name, indexing='position')
#
#    @property
#    def axes(self):
#	return read_dimensions(self.f, name=self.name)
#
#    def __repr__(self):
#	return "\n".join(["{}: {}".format(self.__class__.__name__, self.name),repr(self.axes)])
