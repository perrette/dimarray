""" tools shared by the different variations of a labelled array
"""
import numpy as np

def _operation(func, o1, o2, align=True, order=None):
    """ operation on LaxArray objects

    input:
	func	: operator
	o1    	: LHS operand: expect attributes (values, names)
		  and method conform_to 
	o2    	: RHS operand: at least: be convertible by np.array())
	align, optional: if True, use pandas to align the axes
	order	: if order is True, align the dimensions along a particular order

	_constructor: constructor (takes values and names)

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

    return _constructor(res, order)

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

def _convert_dtype(dtype):
    """ convert a python type
    """
    if dtype is np.dtype(int):
	type_ = int

    else:
	type_ = float

    return type_


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
