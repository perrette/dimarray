#import numpy as np
#import functools
#import inspect
#from lazyapi import axis as Axis
import numpy as np
from functools import partial

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


#def _check_scalar(values):
#    """ export to python scalar if size == 1
#    """
#    avalues = np.array(values, copy=False)
#    
#    if avalues.size == 1:
#	type_ = _convert_dtype(avalues.dtype)
#	result = type_(avalues)
#	test_scalar = True
#    else:
#	result = values
#	test_scalar = False
#    return result, test_scalar



#
# Recursively apply a transformation on an Dimarray
#
# ... decorator to make a function recursive
# 
def _DescRecursive(object):
    """ Decorator to make a Dimarray method recursive via apply_recursive
    """
    def __init__(self, trans, dims):

	assert type(dims) is tuple, "dims must be a tuple"
	self.trans = trans
	self.dims = dims # dimensions to which it applies

    def __get__(self, obj, cls=None):
	""" 
	"""
	newmethod = partial(self.apply, obj, self.dims, self.trans)
	newmethod.__doc__ = "Wrapper for recursive application of this function\n\n"+self.apply_recursive.__doc__
	return newmethod

    return 

def apply_reduce(obj, fun, axis=None, args=(), **kwargs):
    """ Apply a transformation which reduces the axis
    """
    if axis is None:
	axis = obj.axes[0].name # first axis by default (could change that)

    myaxis = obj.axes[axis] # return an Axis object

    # get the axis integer index
    idx = obj.dims.index(myaxis.name)

    # the result as numpy array
    data = fun(obj.values, axis=idx, *args, **kwargs)

    newaxes = [ax for ax in obj.axes if ax.name != myaxis.name]

    # Back to Dimarray
    new = obj._dimarray_api(data, newaxes)

    # add history or the transformation to the attribute?
    _add_history(new, fun, obj.axes[axis])

    return new

def _add_history(new, fun, axis):
    """ document the transformation
    """
    pass
    #if not hasattr(new, 'history'):
    #    new.history = ""
    #new.history += "{}: {},".format(fun.__name__, axis)

    # percentiles return a list in some situations: add a leading "sample" dimension
    if type(data) is list:
	newaxes = [ax for ax in obj.axes if ax.name != myaxis.name]
	q = args[0] # second argument for percentile
	newlab = "pct_"+axis
	newaxes = [Axis(q, newlab)] + newaxes

    # new dimensions (after cumulating on the axis)
    elif data.shape == obj.shape:
	newaxes = obj.axes  # same axis 

    # new dimensions (after collapse of an axis)
    else:
	newaxes = [ax for ax in obj.axes if ax.name != myaxis.name]

    # Back to Dimarray
    new = obj._dimarray_api(data, newaxes)

    # could also add history...
    if False:
	if not hasattr(new, 'history'):
	    new.history = ""
	new.history += "{}: {},".format(fun.__name__, obj.axes[axis])

    return new


#
#   That is it: below are more advanced stuff which are not yet used
#

#
# technicality: lookup for appropriate functions in a global()
#
def search_fun(first=None, has=None, hasnot=None, glob=None):
    """ search function according to its call arguments
    """
    if glob is None:
	glob = globals()

    funs = []
    for funname in glob:
	if funname.startswith("_"): continue
	fun = glob[funname]
	if not inspect.isfunction(fun): continue

	# check arguments
	args = inspect.getargspec(fun)[0]

	# check the function as certain arguments
	if has is not None:
	    if not set(has).issubset(args):
		continue

	# ...and not others
	if hasnot is not None:
	    if set(hasnot).issubset(args):
		continue

	# and check the first arguments
	if first is not None:
	    if not set(first).issubset(args[:len(first)]):
		continue

	funs.append(fun)

    return funs

#
#
#   Decorator to make a bound method from a function with such a signature:
#
#   fun(dim1, ..., dimn, values, *args, **kwargs)
#
def bound(fun):
    """ Decorator

    original function signature:
	fun(dim1, dim2,..., dimn, values, *args, **kwargs)

    new function signature:
	fun(obj, *args, **kwargs)
    """
    # get dimensions from function's signature
    dims = _get_dims(fun)

    def boundmethod(obj, *args, **kwargs):
	""" bound method
	"""
	args0 = [obj.axes[k].values for k in dims] # coordinates: e.g. lon, lat...
	args0.append(obj.values)    # add values
	args1 = args0 + list(args)  # add other default arguments

	return fun(*args1, **kwargs)

    boundmethod.__name__ += fun.__name__
    boundmethod.__doc__ += fun.__doc__

    return boundmethod

#
# Useful for plotting methods
# 
class DimDesc(object):
    """ Choose between a set of methods based on dimensions
    """
    def __init__(self, **pool):
	""" pool: dictionary {dimensions : method}
	"""
	self.pool = pool

    def __get__(self, obj, cls=None):
	"""
	"""
	for k in self.pool:
	    dims = tuple(k.split(","))
	    if dims == tuple(obj.dims):
		fun = self.pool[k]
		return functools.partial(fun, obj)

	raise NotImplementedError("method not implemented for this set of dimensions: "+repr(obj.dims))

#
#
# Descriptor to apply a transformation recursively until the appropriate sub-array is found
# 
#   A transformation compatible with this descriptor looks like:
#
#   fun(dim1, ..., dimn, values, *args, **kwargs)
#
class RecDesc(object):
    """ Recursively apply a transformation whose signature is:

    fun(dim1, ..., dimn, values, *args, **kwargs) returns Dimarray's instance
    """
    def __init__(self, transform):
	""" 
	transform   : transformation to apply (from transform.py) (check above for signature)
	"""
	self.transform = transform 

    def __get__(self, obj, cls=None): 
	""" for python to recognize this as a method
	"""
	# get dimensions from function's signature
	dims = _get_dims(self.transform)
	boundmethod = bound(self.transform)

	# make the function recursive: applied until dims match dim1, etc...
	recmethod = recursive(dims)(boundmethod) 

	return functools.partial(boundmethod, obj)

def _get_dims(fun):
    """ return the dimensions of a functions based on its call signature

    input:
	function with arguments: dim1, dim2..., dimn, values, *args, **kwargs

    returns:
	[dim1, dim2, ..., dimn]
    """
    args = inspect.getargspec(fun)[0] # e.g. lon, lat, values
    i = args.index('values')
    return args[:i] # all cut-off values (e.g. lon, lat) 

#
# Recursively apply a transformation on an Dimarray
#
# ... decorator to make a function recursive
# 
def recursive(dims):
    """ Decorator to make an Dimarray method recursive via apply_recursive
    """
    def real_decorator(fun):
	""" nested decorator to pass "dims" argument
	"""
	def rec_func(obj, *args, **kwargs):
	    """ recursive function
	    """
	    return _apply_recursive(obj, dims, fun, *args, **kwargs)
	return rec_func

    return real_decorator
# 
#
# ... which makes use of:
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
    new = obj._dimarray_api(data, newaxes)

    #new.sort(inplace=True, order=self.dims) # just in case

    return new

