""" collection of base obeje
"""
from collections import OrderedDict as odict
import numpy as np

from core import DimArray, array, Axis, Axes
from core import align_axes 
from core import pandas_obj

def _get_list_arrays(data, keys):
    """ initialize from DimArray objects (internal method)
    """
    if not isinstance(data, dict) and not isinstance(data, list):
	raise TypeError("Type not understood. Expected list or dict, got {}: {}".format(type(data), data))

    assert keys is None or len(keys) == len(data), "keys do not match with data length !"

    # Transform data to a list
    if isinstance(data, dict):
	if keys is None:
	    keys = data.keys()
	data = [data[k] for k in keys]
    
    # Check everything is a DimArray
    for v in data:
	if not isinstance(v, DimArray):
	    raise TypeError("A Dataset can only store DimArray instances")

    # Check names
    for i, v in enumerate(data):
	if not hasattr(v,'name') or not v.name:
	    if keys is not None:
		v.name = keys[i]
	    else:
		v.name = "v%i"%(i)

    return data

class Dataset(object):
    """ Container for a set of aligned objects
    """
    def __init__(self, data=None, keys=None, axes=None):
	""" initialize a dataset from a set of objects of varying dimensions

	data  : dict of DimArrays or list of named DimArrays or Axes object
	keys  : keys to order data if provided as dict, or to name data if list
	"""
	assert data is None or axes is None, "can't provide both data and axes"
	assert keys is None or type(keys) in (list, np.ndarray, tuple) and np.isscalar(keys[0]), "pb with keys"

	# Basic initialization
	if axes is None:
	    axes = Axes()
	else:
	    axes = Axes._init(axes)
	self.axes = axes
	self.variables = odict()

	# If initialized from data
	if data is not None:

	    # First, get a list of named DataFrames
	    data = _get_list_arrays(data, keys)

	    # Align objects
	    data = align_axes(*data)

	    # Append object (will automatically update self.axes)
	    for v in data:
		self[v.name] = v

    @property
    def dimensions(self):
	""" dimensions present in the Dataset as a dictionary
	"""
	dims = odict()
	for ax in self.axes:
	    dims[ax.name] = ax.size
	return dims

    def update(self, dict_):
	""" update from another dataset or dictionary

	"""
	for k in dict_:
	    self[k] = dict_[k]
	
    def __getattr__(self, att):
	"""
	"""
	return getattr(self.variables, att)

    def __repr__(self):
	""" string representation
	"""
	lines = []
	header = "Dataset of %s variables" % (len(self))
	if len(self) == 1: header = header.replace('variables','variable')
	lines.append(header)
	axes = repr(self.axes)
	lines.append(axes)
	for nm in self.variables:
	    dims = self.variables[nm].dims
	    shape = self.variables[nm].shape
	    #print nm,":", ", ".join(dims)
	    lines.append("{}: {}".format(nm,", ".join(dims)))
	return "\n".join(lines)

    def __getitem__(self, item):
	""" 
	"""
	return self.variables[item]

    def __delitem__(self, item):
	""" 
	"""
	axes = self.variables[item].axes
	del self.variables[item]

	# update axes
	for ax in axes:
	    found = False
	    for k in self:
		if ax.name in self[k].dims:
		    found = True
		    continue
	    if not found:
		self.axes.remove(ax)

    def __len__(self):
	return len(self.variables)

    def __iter__(self):
	return iter(self.variables)

    def __setitem__(self, key, val):
	""" Make sure the object is a DimArray with appropriate axes

	Tests:
	-----
	>>> axes = da.Axes.from_tuples(('time',[1, 2, 3]))
	>>> ds = Dataset(axes=axes)
	>>> ds
	Dataset of 0 variables
	dimensions: 'time'
	0 / time (3): 1 to 3
	>>> a = DimArray([0, 1, 2], dims=('time',))
	>>> ds['yo'] = a # doctest: +SKIP
	ValueError: axes values do not match, align data first.			    
	Dataset: time(3)=1:3, 
	Got: time(3)=0:2
	>>> ds['yo'] = a.reindex_like(ds)
	>>> ds['yo']
	dimarray: 2 non-null elements (1 null)
	dimensions: 'time'
	0 / time (3): 1 to 3
	array([  1.,   2.,  nan])
	"""
	if not isinstance(val, DimArray):
	    raise TypeError("can only append DimArray instances")

	if not np.isscalar(key):
	    raise TypeError("only scalar keys allowed")

	# Check dimensions
	for axis in val.axes:

	    # Check dimensions if already existing axis
	    if axis.name in [ax.name for ax in self.axes]:
		if not axis == self.axes[axis.name]:
		    raise ValueError("axes values do not match, align data first.\
			    \nDataset: {}, \nGot: {}".format(self.axes[axis.name], axis))

	    # Append new axis
	    else:
		self.axes.append(axis)

	# update name
	val.name = key
	self.variables[key] = val

    def write(self, f, *args, **kwargs):
	import io.nc as ncio
	ncio.write_dataset(f, self, *args, **kwargs)

    @classmethod
    def read(cls, f, *args, **kwargs):
	import io.nc as ncio
	return ncio.read_dataset(f, *args, **kwargs)

    def to_array(self, axis="items"):
	""" Convert to DimArray

	axis  : axis name, by default "items"

	NOTE: will raise an error if axis name already present in the data
	"""
	#if names is not None or dims is not None:
	#    return self.subset(names=names, dims=dims).to_array()

	if axis in self.dimensions.keys():
	    raise ValueError("please provide an axis name which does not \
		    already exist in Dataset")

	# align all variables to the same dimensions
	data = odict()
	for k in self:
	    data[k] = self[k].reshape(self.dimensions.keys()).broadcast(self.axes)

	# make it a numpy array
	data = [data[k].values for k in self]
	data = np.array(data)

	# determine axes
	axes = [Axis(self.keys(), axis)] + self.axes 

	return DimArray(data, axes)

    def subset(self, names=None, dims=None):
	""" return a subset of the dictionary

	names: variable names
	dims : dimensions to conform too
	"""
	if names is None and dims is not None:
	    names = self._filter_dims(dims)

	d = self.__class__()
	for nm in names:
	    d[nm] = self[nm]
	return d

    def _filter_dims(self, dims):
	""" return variables names matching given dimensions
	"""
	nms = []
	for nm in self:
	    if tuple(self[nm].dims) == tuple(dims):
		nms.append(nm)
	return nms

