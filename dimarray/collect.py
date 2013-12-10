""" collection of base obeje
"""
from collections import OrderedDict
import numpy as np

from core import DimArray
from lazyapi import array, axis, pandas_obj

class Dataset(OrderedDict):
    """ container for a mixture of objects of different types (different dimensions)
    """
    def __setitem__(self, item, val):
	""" convert the object with dimarray
	"""
	if not isinstance(val, DimArray):
	    val = array(val)

	# update the name
	if not val.name:
	    val.name = item

	# Ordered-Dict method
	super(Dataset, self).__setitem__(item, val)


    def apply(method, *args, **kwargs):
	""" apply a dimarray method to every element...
	"""
	d = Dataset()
	for k in self:
	    res = getattr(self[k], method)(*args, **kwargs)

	    # reinsert in the dictionary (typically False for plots, true for transformations)
	    if isinstance(res, DimArray): d[k] = res

	return d

    def save(self, f, *args, **kwargs):
	import ncio
	ncio.write_dataset(f, self, *args, **kwargs)

    @classmethod
    def load(cls, f, *args, **kwargs):
	import ncio
	return ncio.read_dataset(f, *args, **kwargs)

    def align(self):
	""" align the data using pandas

	note: assume consistent objects
	"""
	# Transform each item to a pandas object
	d = OrderedDict()
	for k in self:
	    n = self[k].ndim
	    d[k] = self[k].to_pandas()
	
	# Now transform the whole dictionary to a pandas object
	d = pandas_obj(d)

	# Transform back to Dataset object
	cls = d[k].__class__
	data = Dataset()
	for k in self:
	    data[k] = cls(d[k].values, d[k].axes)

	return data

    def isaligned(self):
	""" check that axes are consistent
	"""
	axes = self.values()[0].axes
	for k in self:
	    if len(self[k].axes) != len(axes):
		return False
	    for i, ax in enumerate(self[k].axes):
		if not np.all(ax.values == axes[i].values):
		    return False

	return True

    def to_dimarray(self, nms=None, dims=None):
	""" return a set variable (can filter according to names or dimensions)
	"""
	if nms is not None or dims is not None:
	    return self.subset(nms=nms, dims=dims).to_dimarray()

	# align dimensions
	if not self.isaligned():
	    return self.align().to_dimarray()

	# make it a numpy array
	data = [self[k].values for k in self]
	data = np.array(data)

	# determine axes
	v0 = self.values()[0]
	axes = [axis(self.keys(), "item")] + v0.axes 

	#cls = base.get_class(('set',) + v0._dimensions) # get class

	return array(data, axes)

    def _filter_dims(self, dims):
	""" return variables names matching given dimensions
	"""
	nms = []
	for nm in self:
	    if tuple(self[nm].dims) == tuple(dims):
		nms.append(nm)
	return nms

    def subset(self, nms=None, dims=None):
	""" return a subset of the dictionary
	"""
	if nms is None and dims is not None:
	    nms = self._filter_dims(dims)

	d = self.__class__()
	for nm in nms:
	    d[nm] = self[nm]
	return d

    def __repr__(self):
	""" string representation
	"""
	header = "Dataset of %s variables" % (len(self))
	#content = "\n".join(["%s : %s" % (k, repr(self[k])) for k in self])
	content = "\n\n".join([repr(self[k]) for k in self])
	return "\n\n".join([header, content])

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

