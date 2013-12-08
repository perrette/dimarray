""" Basic operation between a dimarrays and something else
"""

def _operation(func, o1, o2, align=True, order=None):
    """ operation on LaxArray objects

    input:
	func	: operator
	o1    	: LHS operand: expect attributes (values, dims)
		  and method conform_to 
	o2    	: RHS operand: at least: be convertible by np.array())
	align, optional: if True, use pandas to align the axes
	order	: if order is True, align the dimensions along a particular order

    output:
	values: array values
	dims : dimension names
    """
    # second operand is not a DimArray: let numpy do the job 
    if not hasattr(o2, 'values') or not hasattr(o2,'dims'): 
	if np.ndim(o2) > o1.ndim:
	    raise ValueError("bad input: second operand's dimensions not documented")
	res = func(o1.values, np.array(o2))
	return res, o1.dims

    # if same dimension: easy
    if o1.dims == o2.dims:
	res = func(o1.values, o2.values)
	return res, o1.dims

    # otherwise determine the dimensions of the result
    if order is None:
	order = _unique(o1.dims + o2.dims) 
    else:
	order = [o for o in order if o in o1.dims or o in o2.dims]
    order = tuple(order)

    #
    o1 = o1.conform_to(order)
    o2 = o2.conform_to(order)

    assert o1.dims == o2.dims, "problem in transpose"
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

