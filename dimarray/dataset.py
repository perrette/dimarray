""" collection of base obeje
"""
from collections import OrderedDict as odict
import numpy as np

from core import DimArray, array, Axis, Axes
from core import align_axes 
from core import pandas_obj

class Dataset(object):
    """ Container for a set of aligned objects
    """
    def __init__(self, data, keys=None):
	""" initialize a dataset from a set of objects of varying dimensions

	data  : dict of DimArrays or list of named DimArrays
	keys  : keys to order data if provided as dict, or to name data if list
	"""
	# Initialize an Axes object
	self.axes = Axes()
	self.variables = odict()

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
	    if keys is not None:
		v.name = keys[i]

	    if not v.name:
		raise ValueError("DimArray must have a name if provided as list, provide `keys=` parameter")

	# Align objects
	data = align_axes(*data)

	# Append to dictionary
	for v in data:
	    self[v.name] = v

#    @classmethod
#    def from_dict(cls, dict_, keys=()):
#	""" initialize from dictionary
#	"""
#	# use keys to order the dictionary
#	args = []
#	for k in keys:
#	    data = dict_.pop(k)
#	    data.name = k
#	    args.append(data)
#
#	return cls(*args, **dict_)
#     
#    @classmethod
#    def from_arrays(cls, ):
#	""" initialize from dictionary
#	"""
#	pass

    @property
    def dimensions(self):
	""" dimensions present in the Dataset as a dictionary
	"""
	dims = odict()
	for ax in self.axes:
	    dims[ax.name] = ax.size
	return dims

    def update(self, odict):
	""" update from another dataset or dictionary
	"""
	for k in enumerate(odict):
	    self[k] = odict[k]
	
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

    def __len__(self):
	return len(self.variables)

    def __iter__(self):
	return iter(self.variables)

    def __setitem__(self, item, val):
	""" Make sure the object is a DimArray with appropriate axes
	"""
	if not isinstance(val, DimArray):
	    raise TypeError("can only append DimArray instances")

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
	val.name = item
	self.variables[item] = val

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

