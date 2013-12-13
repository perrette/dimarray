""" collection of base obeje
"""
from collections import OrderedDict as odict
import numpy as np

from axes import Axis, Axes
from core import Dimarray, array, _align_objects
from tools import pandas_obj

def _check_args(args=(), **kwargs):
    """ check input arguments for a Dataset object, return as list
    """
    # Check everything is a Dimarray
    variables = args+kwargs.values()
    for v in variables:
	if not isinstance(v, Dimarray):
	    raise TypeError("A Dataset can only store Dimarray instances")

    # Check args have a name
    for v in args:
	if not v.name:
	    raise ValueError("Dimarray must have a name if provided as list, try using key-word arguments")

    # Update the variable names
    for k in kwargs:
	kwargs[k].name = k

    return args+kwargs.values()


class Dataset(object):
    """ Container for a set of aligned objects
    """
    def __init__(self, *args, **kwargs):
	""" initialize a dataset from a set of objects of varying dimensions

	args  : variable list of Dimarrays (must have a name)
	kwargs: dimarrays as keyword arguments
	"""
	# Initialize an Axes object
	self.axes = Axes()
	self.variables = odict()

	# Make basic checks and return a list of named variables
	variables = _check_args(args, **kwargs)

	# Align objects
	variables = _align_objects(variables)

	# Append to dictionary
	for i, v in enumerate(variables):
	    self[k] = v

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
	return self.variables.__getattr__(att)

    def __repr__(self):
	""" string representation
	"""
	lines = []
	header = "Dataset of %s variables" % (len(self))
	lines.append(header)
	axes = repr(self.axes)
	lines.append(axes)
	for nm in self.variables:
	    dims = self.variables[nm].dims
	    shape = self.variables[nm].shape
	    print nm,":", ", ".join(dims)
	    lines.append("{}: {}".format(", ".join(dims)))
	return "\n".join(lines)

    def __getitem__(self, item):
	""" 
	"""
	return self.variables[item]

    def __setitem__(self, item, val):
	""" Make sure the object is a Dimarray with appropriate axes
	"""
	if not isinstance(val, Dimarray):
	    raise TypeError("can only append Dimarray instances")

	# Check dimensions
	for axis in item.axes:

	    # Check dimensions if already existing axis
	    if axis.name in self.axes:
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
	import ncio
	ncio.write_dataset(f, self, *args, **kwargs)

    @classmethod
    def read(cls, f, *args, **kwargs):
	import ncio
	return ncio.read_dataset(f, *args, **kwargs)

    def to_array(self):
	""" Convert to Dimarray
	"""
	#if names is not None or dims is not None:
	#    return self.subset(names=names, dims=dims).to_array()

	# align all variables to the same dimensions
	data = odict()
	for k in self:
	    data[k] = self.reshape(self.dimensions.keys()).expand(*self.axes)

	# make it a numpy array
	data = [self[k].values for k in self]
	data = np.array(data)

	# determine axes
	v0 = self.values()[0]
	axes = [axis(self.keys(), "item")] + v0.axes 

	#cls = base.get_class(('set',) + v0._dimensions) # get class

	return array(data, axes)

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

    def plots(self, axes=None, **kwargs):
	""" Call the underlying methods: Individual plots in separate figures: 
	"""
	import matplotlib.pyplot as plt
	if axes is None:
	    axes = [plt.figure().gca() for k in self]

	hs = []
	for i, k in enumerate(self):
	    ax = axes.flatten()[i]
	    h = self[k].plot(ax=ax, **kwargs)
	    ax.set_title(k)
	    hs.append(h)

	return hs

    def plot(self, ni=None, nj=None, kwargs={}, **subplot):
	""" All subplots in the same one Figure
	"""
	from plotting import get_grid
	import matplotlib.pyplot as plt

	if ni is None or nj is None:
	    n = len(self)
	    ni, nj = get_grid(n)

	fig, axes = plt.subplots(ni, nj, squeeze=False, **subplot)

	return self.plots(axes, **kwargs)

