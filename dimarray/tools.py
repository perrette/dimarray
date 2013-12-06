""" tools shared by the different variations of a labelled array
"""
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


def _operation(func, o1, o2, align=True, order=None):
    """ operation on LaxArray objects

    input:
	func	: operator
	o1    	: LHS operand: expect attributes (values, names)
		  and method conform_to 
	o2    	: RHS operand: at least: be convertible by np.array())
	align, optional: if True, use pandas to align the axes
	order	: if order is True, align the dimensions along a particular order

    output:
	values: array values
	names : dimension names
    """
    # second operand is not a LaxArray: let numpy do the job 
    if not hasattr(o2, 'values') or not hasattr(o2,'names'): 
	if np.ndim(o2) > o1.ndim:
	    raise ValueError("bad input: second operand's dimensions not documented")
	res = func(o1.values, np.array(o2))
	return res, o1.names

    # if same dimension: easy
    if o1.names == o2.names:
	res = func(o1.values, o2.values)
	return res, o1.names

    # otherwise determine the dimensions of the result
    if order is None:
	order = _unique(o1.names + o2.names) 
    else:
	order = [o for o in order if o in o1.names or o in o2.names]
    order = tuple(order)

    #
    o1 = o1.conform_to(order)
    o2 = o2.conform_to(order)

    assert o1.names == o2.names, "problem in transpose"
    res = func(o1.values, o2.values) # just to the job

    return res, order

#
# Handle operations 
#
def _unique(nm):
    """ return the same ordered list without duplicated elements
    """
    new = []
    for k in nm:
	if k not in new:
	    new.append(k)
    return new

def _check_scalar(values):
    """ export to python scalar if size == 1
    """
    avalues = np.array(values, copy=False)
    
    if avalues.size == 1:
	type_ = _convert_dtype(avalues.dtype)
	result = type_(avalues)
	test_scalar = True
    else:
	result = values
	test_scalar = False
    return result, test_scalar



def _slice_to_indices(slice_, n, include_last=False, bounds=None):
    """ convert a slice into indices for an array or list of size n

    input:
	slice_: slice object
	n     : size of the array
	include_last: include last index?
	bounds: TO DO

    output:
	array of indices
    """
    if bounds is not None:
	raise NotImplementedError("bound checking not yet implemented !")

    if include_last:

	# might need more checks
	if type(slice_.stop) is int:
	    stop = slice_.stop+1
	else:
	    stop = slice_.stop

	# new slice
	slice_ = slice(slice_.start, stop, slice_.step) 

    # convertion to indices 
    idx = np.arange(n) # array to sample (by default same as arange(n)
    indices = idx[slice_]


    return indices

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


#def deco_numpy(apply_method, numpy_method, **kwargs):
#    """ predefine arguments for apply
#
#    input:
#	apply_method: method to "decorate" (which is BaseArray.apply)
#	numpy_method: first argument to apply_method
#	**kwargs    : default arguments for the numpy method
#    output:
#	decorated apply_method
#
#    NOTE: could also add *args parameter if the need comes, here not included
#          to reduce confusion
#    
#    decorator similar to partial, but expand args to *args
#
#    Examples input:
#	function: apply(fun, axis=0, skipna=True, args=(), **kwargs)
#
#    Corresponding output
#	function: apply(*args, axis=0, skipna=True, fun=fun, **kwargs)
#    """
#    def newfun(*varargs, **kwds):
#	""" the function actually called when doing, e.g. a.mean()
#	""" 
#	kwargs.update(kwds)
#	# here is the trick: pass args to args parameter instead of *args
#	return apply_method(numpy_method, args=varargs, **kwargs)
#
#    # Now replace the doc string with "apply" docstring
#    newfun.__doc__ = apply_method.__doc__.replace("`func`","`"+numpy_method+"`")
#
#    return newfun
