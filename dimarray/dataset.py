""" collection of base obeje
"""
from collections import OrderedDict as odict
import numpy as np

from core import DimArray, array, Axis, Axes
from core import align_axes 
from core import pandas_obj
from core.metadata import MetadataDesc

class Dataset(odict):
    """ Container for a set of aligned objects
    """

    _metadata = MetadataDesc(exclude=['axes'])

    def __init__(self, *args, **kwargs):
	""" initialize a dataset from a set of objects of varying dimensions

	data  : dict of DimArrays or list of named DimArrays or Axes object
	keys  : keys to order data if provided as dict, or to name data if list
	"""
	assert not {'axes','keys'}.issubset(kwargs.keys()) # just to check bugs due to back-compat ==> TO BE REMOVED AFTER DEBUGGING

	# check input arguments: same init as odict
	kwargs = odict(*args, **kwargs)

	# initialize an ordered dictionary
	super(Dataset, self).__init__()

	# Basic initialization
	self.axes = Axes()

	values = kwargs.values()
	keys = kwargs.keys()

        # Check everything is a DimArray
	for key, value in zip(keys, values):
	    if not isinstance(value, DimArray):
		raise TypeError("A Dataset can only store DimArray instances, got {}: {}".format(key, value))

	# Align objects
	values = align_axes(*values)

	# Append object (will automatically update self.axes)
	for key, value in zip(keys, values):
	    self[key] = value

    @property
    def dims(self):
	""" list of dimensions contained in the Dataset, consistently with DimArray's `dims`
	"""
	return [ax.name for ax in self.axes]

    def __repr__(self):
	""" string representation
	"""
	lines = []
	header = "Dataset of %s variables" % (len(self))
	if len(self) == 1: header = header.replace('variables','variable')
	lines.append(header)
	axes = repr(self.axes)
	lines.append(axes)
	for nm in self.keys():
	    dims = self[nm].dims
	    shape = self[nm].shape
	    #print nm,":", ", ".join(dims)
	    lines.append("{}: {}".format(nm,repr(dims)))
	return "\n".join(lines)

    def __delitem__(self, item):
	""" 
	"""
	axes = self[item].axes
	super(Dataset, self).__delitem__(item)

	# update axes
	for ax in axes:
	    found = False
	    for k in self:
		if ax.name in self[k].dims:
		    found = True
		    continue
	    if not found:
		self.axes.remove(ax)

    def __setitem__(self, key, val):
	""" Make sure the object is a DimArray with appropriate axes

	Tests:
	-----
#	>>> axes = da.Axes.from_tuples(('time',[1, 2, 3]))
	>>> ds = Dataset()
	>>> ds
	Dataset of 0 variables
	dimensions: 
	<BLANKLINE>
	>>> a = DimArray([0, 1, 2], dims=('time',))
	>>> ds['yo'] = a 
	>>> ds['yo']
	dimarray: 3 non-null elements (0 null)
	dimensions: 'time'
	0 / time (3): 0 to 2
	array([0, 1, 2])
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
	super(Dataset, self).__setitem__(key, val)

    def write_nc(self, f, *args, **kwargs):
	import io.nc as ncio
	ncio.write_dataset(f, self, *args, **kwargs)

    write = write_nc

    @classmethod
    def read_nc(cls, f, *args, **kwargs):
	import io.nc as ncio
	return ncio.read_dataset(f, *args, **kwargs)

    read = read_nc

    def to_array(self, axis=None, keys=None, _constructor=None):
	""" Convert to DimArray

	axis  : axis name, by default "unnamed"
	"""
	#if names is not None or dims is not None:
	#    return self.subset(names=names, dims=dims).to_array()

	if axis is None:
	    axis = "unnamed"
	    if axis in self.dims:
		i = 1
		while "unnamed_{}".format(i) in self.dims:
		    i+=1
		axis = "unnamed_{}".format(i)

	if axis in self.dims:
	    raise ValueError("please provide an axis name which does not \
		    already exist in Dataset")

	if keys is None:
	    keys = self.keys()

	# align all variables to the same dimensions
	data = odict()

	for k in keys:
	    data[k] = self[k].reshape(self.dims).broadcast(self.axes)

	# make it a numpy array
	data = [data[k].values for k in keys]
	data = np.array(data)

	# determine axes
	axes = [Axis(keys, axis)] + self.axes 

	if _constructor is None: _constructor = DimArray
	return _constructor(data, axes)

#    def subset(self, names=None, dims=None):
#	""" return a subset of the dictionary
#
#	names: variable names
#	dims : dimensions to conform too
#	"""
#	if names is None and dims is not None:
#	    names = self._filter_dims(dims)
#
#	d = self.__class__()
#	for nm in names:
#	    d[nm] = self[nm]
#	return d
#
#    def _filter_dims(self, dims):
#	""" return variables names matching given dimensions
#	"""
#	nms = []
#	for nm in self:
#	    if tuple(self[nm].dims) == tuple(dims):
#		nms.append(nm)
#	return nms
#

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
	else:
	    assert set(keys) == set(data.keys()), "keys do not match dictionary keys"
	data = [data[k] for k in keys]
    
    # Check everything is a DimArray
    for v in data:
	if not isinstance(v, DimArray):
	    raise TypeError("A Dataset can only store DimArray instances")

    # Assign names
    if keys is not None:
	for i, v in enumerate(data):
	    v = v.copy(shallow=True) # otherwise name is changed on the caller side
	    v.name = keys[i]
	    data[i] = v

    else:
	for i, v in enumerate(data):
	    if not hasattr(v,'name') or not v.name:
		v.name = "v%i"%(i)

    return data

def test():
    """
    >>> data = test() 
    >>> data['test2'] = da.DimArray([0,3],('source',['greenland','antarctica'])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
	...
    ValueError: axes values do not match, align data first.			    
    Dataset: source(1)=greenland:greenland, 
    Got: source(2)=greenland:antarctica
    >>> data['ts']
    dimarray: 5 non-null elements (5 null)
    dimensions: 'time'
    0 / time (10): 1950 to 1959
    array([  0.,   1.,   2.,   3.,   4.,  nan,  nan,  nan,  nan,  nan])
    >>> data.to_array(axis='items')
    dimarray: 12250 non-null elements (1750 null)
    dimensions: 'items', 'lon', 'lat', 'time', 'source'
    0 / items (4): mymap to test
    1 / lon (50): -180.0 to 180.0
    2 / lat (7): -90.0 to 90.0
    3 / time (10): 1950 to 1959
    4 / source (1): greenland to greenland
    array(...)
    """
    import dimarray as da
    axes = da.Axes.from_tuples(('time',[1, 2, 3]))
    ds = da.Dataset()
    a = da.DimArray([[0, 1],[2, 3]], dims=('time','items'))
    ds['yo'] = a.reindex_like(axes)

    np.random.seed(0)
    mymap = da.DimArray.from_kw(np.random.randn(50,7), lon=np.linspace(-180,180,50), lat=np.linspace(-90,90,7))
    ts = da.DimArray(np.arange(5), ('time',np.arange(1950,1955)))
    ts2 = da.DimArray(np.arange(10), ('time',np.arange(1950,1960)))

    # Define a Dataset made of several variables
    data = da.Dataset({'ts':ts, 'ts2':ts2, 'mymap':mymap})
    #data = da.Dataset([ts, ts2, mymap], keys=['ts','ts2','mymap'])

    assert np.all(data['ts'].time == data['ts2'].time),"Dataset: pb data alignment" 

    data['test'] = da.DimArray([0],('source',['greenland']))  # new axis
    #data

    return data
