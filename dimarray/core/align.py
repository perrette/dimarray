""" Functions associated to array alignment
"""
import numpy as np
import itertools
from axes import Axes, Axis

def broadcast_arrays(*objects):
    """ Analogous to numpy.broadcast_arrays
    
    but with looser requirements on input shape
    and returns copy instead of views

    objects: variable list of DimArrays

    Examples:
    ---------
    Just as numpy's broadcast_arrays:
    >>> x = da.array([[1,2,3]])
    >>> y = da.array([[1],[2],[3]])
    >>> da.broadcast_arrays(x, y)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (3): 0 to 2
    array([[1, 2, 3],
	   [1, 2, 3],
	   [1, 2, 3]]),
     dimarray: 9 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (3): 0 to 2
    array([[1, 1, 1],
	   [2, 2, 2],
	   [3, 3, 3]])]
    """
    # give all objects the same dimension (without changing the size)
    objects = align_dims(*objects)

    # get axes object made of all non-singleton common axes
    try:
	axes = get_axes(*objects)

    # fails if axes are not aligned
    except AssertionError, msg:
	raise ValueError(msg)

    # now broadcast each DimArray along commmon axes
    newobjects = []
    for o in objects:
	o = o.broadcast(axes)
	newobjects.append(o)

    return newobjects



def get_dims(*objects):
    """ find all dimensions from a variable list of objects
    """
    dims = []
    for o in objects:
	for dim in o.dims:
	    if dim not in dims:
		dims.append(dim)

    return dims

def get_axes(*objects):
    """ find list of axes from a list of axis-aligned DimArray objects
    """
    dims = get_dims(*objects) # all dimensions present in objects
    axes = Axes()

    for dim in dims:

	common_axis = None

	for o in objects:

	    # skip missing dimensions
	    if dim not in o.dims: continue

	    axis = o.axes[dim]

	    # update values
	    if common_axis is None or (common_axis.size==1 and axis.size > 1):
		common_axis = axis

	    # Test alignment for non-singleton axes
	    assert axis.size == 1 or np.all(axis.values==common_axis.values), "axes are not aligned"

	# append new axis
	axes.append(common_axis)


    return axes


def align_dims(*objects):
    """ Align dimensions of a list of objects, ready to broadcast

    >>> x = da.DimArray(np.arange(2), 'x0')
    >>> y = da.DimArray(np.arange(3), 'x1')
    >>> da.align_dims(x, y)
    [dimarray: 2 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (1): None to None
    array([[0],
	   [1]]),
     dimarray: 3 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (1): None to None
    1 / x1 (3): 0 to 2
    array([[0, 1, 2]])]
    """
    # If dimensions are already equal, do nothing
    lst = {o.dims for o in objects}
    if len(lst) == 1:
	return objects

    # Determine the dimensions of the result
    newdims = get_dims(*objects) 

    # Reshape all DimArrays
    newobjects = []
    for o in objects:
	o = o.reshape(newdims)
	newobjects.append(o)

    return newobjects


def align_axes(*objects):
    """ align axes of a list of objects by reindexing
    """
    # find the dimensiosn
    dims = get_dims(*objects)

    objects = list(objects)
    for d in dims:

	# objects which have that dimension
	objs = filter(lambda o: d in o.dims, objects)

	# common axis to reindex on
	ax_values = common_axis(*[o.axes[d] for o in objs])

	# update objects
	for i, o in enumerate(objects):
	    if o not in objs:
		continue
	    if o.axes[d] == ax_values:
		continue

	    objects[i] = o.reindex_axis(ax_values, axis=d)

    return objects


def common_axis(*axes):
    """ find the common axis between a list of axes
    """

    # First merge the axes with duplicates (while preserving the order of the lists)
    axes_lists = [list(ax.values) for ax in axes] # axes as lists
    newaxis_val = axes_lists[0]
    for val in itertools.chain(*axes_lists[1:]):
	if val not in newaxis_val:
	    newaxis_val.append(val)

    return Axis(newaxis_val, axes[0].name)


#def aligned(objects, skip_missing=True, skip_singleton=True):
#    """ test whether common non-singleton axes are equal
#    """
#    # check whether all objects have the same dimensions
#    if not skip_missing:
#	set_of_dims = {o.dims for o in objects}
#	if len(set_of_dims) > 1:
#	    return False
#
#    # test whether common non-singleton axes are equal
#    try: 
#	axes = get_axes(*objects)
#    except:
#	return False
#
#    # test whether all existing dimensions have same size
#    if not skip_singleton:
#	dims = [ax.name for ax in axes]
#	for dim in dims:
#	    set_of_sizes = {o.axes[dim].size for o in objects}
#	    if len(set_of_sizes) > 1:
#		return False
#
#    return True