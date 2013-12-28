""" Module to deal with indexing
"""
def xs(self, ix=None, axis=0, method=None, keepdims=False, **axes):
    """ Cross-section, can be multidimensional

    input:

	- ix       : int or list or tuple or slice (index) 
	- axis     : int or str
	- method   : indexing method (default "index")
		     - "numpy": numpy-like integer 
		     - "index": look for exact match similarly to list.index
		     - "nearest": (regular Axis only) nearest match, bound checking
	- keepdims : keep singleton dimensions

	- **axes  : provide axes as keyword arguments for multi-dimensional slicing
		    ==> chained call to xs 
		    Note in this mode axis cannot be named after named keyword arguments
		    (`ix`, `axis`, `method` or `keepdims`)

    output:
	- Dimarray object or python built-in type, consistently with numpy slicing

    >>> a.xs(45.5, axis=0)	 # doctest: +ELLIPSIS
    >>> a.xs(45.7, axis="lat") == a.xs(45.5, axis=0) # "nearest" matching
    True
    >>> a.xs(time=1952.5)
    >>> a.xs(time=70, method="numpy") # 70th element along the time dimension

    >>> a.xs(lon=(30.5, 60.5), lat=45.5) == a[:, 45.5, 30.5:60.5] # multi-indexing, slice...
    True
    >>> a.xs(time=1952, lon=-40, lat=70, method="nearest") # lookup nearest element (with bound checking)
    """
    if method is None:
	method = self._indexing

    # single-axis slicing
    if ix is not None:
	obj = self._xs(ix, axis=axis, method=method, keepdims=keepdims)

    # multi-dimensional slicing <axis name> : <axis index value>
    # just a chained call
    else:
	obj = self
	for nm, idx in axes.iteritems():
	    obj = obj._xs(idx, axis=nm, method=method, keepdims=keepdims)

    return obj


def _xs(self, ix, axis=0, method="index", keepdims=False):
    """ cross-section or slice along a single axis, see xs
    """
    assert axis is not None, "axis= must be provided"

    # get integer index/slice for axis valued index/slice
    if method is None:
	method = self._indexing # slicing method

    # Linear interpolation between axis values, see _take_interp
    if method in ('interp'):
	return self._take_interp(ix, axis=axis, keepdims=keepdims)

    # get an axis object
    ax = self.axes[axis] # axis object
    axis_id = self.axes.index(ax) # corresponding integer index

    # numpy-like indexing, do nothing
    if method == "numpy":
	index = ix

    # otherwise locate the values
    elif method in ('nearest','index'):
	index = ax.loc(ix, method=method) 

    else:
	raise ValueError("Unknown method: "+method)

    # make a numpy index  and use numpy's slice method (`slice(None)` :: `:`)
    index_nd = (slice(None),)*axis_id + (index,)
    newval = self.values[index_nd]
    newaxis = self.axes[axis][index] # returns an Axis object

    # if resulting dimension has reduced, remove the corresponding axis
    axes = copy.copy(self.axes)

    # check for collapsed axis
    collapsed = not isinstance(newaxis, Axis)
	
    # re-expand things even if the axis collapsed
    if collapsed and keepdims:

	newaxis = Axis([newaxis], self.axes[axis].name) 
	reduced_shape = list(self.shape)
	reduced_shape[axis_id] = 1 # reduce to one
	newval = np.reshape(newval, reduced_shape)

	collapsed = False # set as not collapsed

    # If collapsed axis, just remove it and add new stamp
    if collapsed:
	axes.remove(ax)
	stamp = "{}={}".format(ax.name, newaxis)

    # Otherwise just update the axis
    else:
	axes[axis_id] = newaxis
	stamp = None

    # If result is a numpy array, make it a Dimarray
    if isinstance(newval, np.ndarray):
	result = self._constructor(newval, axes, **self._metadata)

	# append stamp
	if stamp: result._metadata_stamp(stamp)

    # otherwise, return scalar
    else:
	result = newval

    return result

def _take_interp(self, ix, axis, keepdims):
    """ Take a number or an integer as from an axis
    """
    assert type(ix) is not slice, "interp only work with integer and list indexing"

    # return a "fractional" index
    ax = self.axes[axis]
    index = ax.loc(ix, method='interp') 
    ii = np.array(index) # make sure it is array-like

    # position indices of nearest neighbors
    # ...last element, 
    i0 = np.array(index, dtype=int)
    i1 = i0+1

    # make sure we are not beyond 
    over = i1 == ax.size
    if i1.size == 1:
	if over:
	    i0 -= 1
	    i1 -= 1
    else:
	i0[over] = ax.size-2
	i1[over] = ax.size-1

    # weight for interpolation
    w1 = ii-i0 

    # sample nearest neighbors
    v0 = self._xs(i0, axis=axis, method="numpy", keepdims=keepdims)
    v1 = self._xs(i1, axis=axis, method="numpy", keepdims=keepdims)

    # result as weighted sum
    values = v0.values*(1-w1) + v1.values*w1
    axes = []
    for d in v0.dims:
	if d == ax.name:
	    axis = Axis(v0.axes[d].values*(1-w1) + v1.axes[d].values*w1, d)
	else:
	    axis = self.axes[d]
	axes.append(axis)
    return self._constructor(values, axes, **self._metadata)

#    def compress(self, condition, axis=None):
#	""" analogous to numpy `compress` method
#	"""
#	return self.apply("compress", axis=axis, skipna=False, args=(condition,))
#
#    def take(self, indices, axis=None):
#	""" analogous to numpy `take` method
#	"""
#	return self.apply("take", axis=axis, skipna=False, args=(indices,))


#    #
#    # add an extra "where" method
#    #
#    def where(self, condition, otherwise=None, axis=None):
#	""" 
#	parameters:
#	-----------
#
#	    condition: bool array of same size as self, unless `axis=` is provided
#		 OR    `str` indicating a condition on axes
#	    otherwise: array of same size as self or scalar, replacement value when condition is False
#	    axis     : if provided, interpret the condition as applying along an axis
#
#	returns:
#	--------
#	    
#	    array with self values when condition is True, and `otherwise` if False
#	    if only `condition` is provided, return axis values for which `condition` is True
#
#	Examples:
#	---------
#	    a.where(a > 0)
#	"""
#	# convert scalar to the right shape
#	if np.size(otherwise) == 1:
#	    otherwise += np.zeros_like(self.values)
#
#	# evaluate str condition
#	if type(condition) is str:
#	    result = eval(condition, {ax.name:ax.values})
#
#	result = np.where(condition, [self.values, otherwise])
