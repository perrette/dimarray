#import numpy as np
#import functools
#import inspect
#from lazyapi import axis as Axis
import numpy as np
from functools import partial
from axes import Axis, Axes

#
# Technicalities to apply numpy methods
#
class _NumpyDesc(object):
    """ to apply a numpy function which reduces an axis
    """
    def __init__(self, apply_method, numpy_method, **kwargs):
	"""
	apply_method: as the string name of the bound method to consider
        numpy_method: as a string name of the numpy function to apply
	**kwargs    : default values for keyword arguments to numpy_method
	"""
	assert type(apply_method) is str, "can only provide method name as a string"
	assert type(numpy_method) is str, "can only provide method name as a string"
	self.numpy_method = numpy_method
	self.apply_method = apply_method
	self.kwargs = kwargs

    def __get__(self, obj, cls=None):
	"""
	"""
	# convert self.apply to an actual function
	apply_method = getattr(obj, self.apply_method) 
	newmethod = partial(apply_method, self.numpy_method, **self.kwargs)
	#newmethod = deco_numpy(apply_method, self.numpy_method, **self.kwargs)

	# Now replace the doc string with "apply" docstring
	newmethod.__doc__ = apply_method.__doc__.replace("`func`","`"+self.numpy_method+"`")

	return newmethod

#
# recursively apply a DimArray ==> DimArray transform
#
def _apply_recursive(obj, dims, fun, *args, **kwargs):
    """ recursively apply a multi-axes transform

    dims   : dimensions to apply the function on
		Recursive call until dimensions match

    fun     : function to (recursively) apply, must returns a Dimarray instance or something compatible
    *args, **kwargs: arguments passed to fun in addition to the axes
    """
    #
    # Check that dimensions are fine
    # 
    assert set(dims).issubset(set(obj.dims)), \
	    "dimensions ({}) not found in the object ({})!".format(dims, obj.dims)

    #
    # If dims exactly matches: Done !
    # 
    if set(obj.dims) == set(dims):
	#assert set(obj.dims) == set(dims), "something went wrong"
	return fun(obj, *args, **kwargs)

    #
    # otherwise recursive call
    #

    # search for an axis which is not in dims
    i = 0
    while obj.axes[0].name in dims:
	i+=1 

    # make sure it worked
    assert obj.axes[i].name not in dims, "something went wrong"

    # now make a slice along this axis
    axis = obj.axes[i]

    # Loop over one axis
    data = []
    for axisval in axis.values: # first axis

	# take a slice of the data (exact matching)
	slice_ = obj.xs(**{ax.name:axisval, "method":"index"})

	# apply the transform recursively
	res = _apply_recursive(slice_, dims, fun, *args, **kwargs)

	data.append(res.values)

    newaxes = [axis] + res.axes # new axes
    data = np.array(data) # numpy array

    # automatically sort in the appropriate order
    new = obj._constructor(data, newaxes)

    #new.sort(inplace=True, order=self.dims) # just in case

    return new

#
# INTERPOLATION
#

def interp1d_numpy(obj, newaxis, axis=None, left=None, right=None):
    """ interpolate along one axis: wrapper around numpy's interp
    """
    ax = Axis(newaxis, axis)

    def interp1d(obj, order=1):
	""" 2-D interpolation function appled recursively on the object
	"""
	xp = obj.axes[x.name].values
	fp = obj.values
	f = np.interp(newaxis.values, xp, fp, left=None, right=None)
	res = interp(obj.values, x0, y0, x1, y1, order=order)
	return obj._constructor(res, newaxes, **obj._metadata)

    return _apply_recursive(obj, dims=(x.name, y.name), interp1d)


def interp2d(obj, newaxes, axes=None, order=1):
    """ bilinear interpolation: wrapper around mpl_toolkits.basemap.interp

    input:
	newaxes: list of Axis object, or list of 1d arrays
	axes, optional: list of str (axis names), required if newaxes is a list of arrays

    output:
	interpolated data (n-d)
    """
    from mpl_toolkits.basemap import interp

    # make sure input axes have the valid format
    newaxes = Axes.from_list(newaxes, axes) # valid format
    newaxes.sort(self.dims) # re-order according to object's dimensions
    x, y = newaxes  # 2-d interpolation

    # make new grid 2-D
    x1, x1 = np.meshgrid(x.values, y.values, indexing='ij')

    def interp2d(obj, order=1):
	""" 2-D interpolation function appled recursively on the object
	"""
	x0, y0 = obj.axes[x.name].values, obj.axes[y.name].values
	res = interp(obj.values, x0, y0, x1, y1, order=order)
	return obj._constructor(res, newaxes, **obj._metadata)

    return _apply_recursive(obj, dims=(x.name, y.name), interp2d)
