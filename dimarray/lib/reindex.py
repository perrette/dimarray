""" Functions associated to array indexing (axis values)
"""
import numpy as np
import itertools

from dimarray.axes import Axes, Axis
from reshape import get_dims
from transform import apply_recursive

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


#
# INTERPOLATION
#

def interp1d_numpy(obj, values=None, axis=0, left=None, right=None):
    """ interpolate along one axis: wrapper around numpy's interp

    input:
	obj : Dimarray
	newaxis_values: 1d array, or Axis object
	newaxis_name, optional: `str` (axis name), required if newaxis is an array

    output:
	interpolated data (n-d)
    """
    newaxis = Axis(values, axis)

    def interp1d(obj, order=1):
	""" 2-D interpolation function appled recursively on the object
	"""
	xp = obj.axes[newaxis.name].values
	fp = obj.values
	result = np.interp(newaxis.values, xp, fp, left=left, right=right)
	return obj._constructor(result, [newaxis], **obj._metadata)

    result = apply_recursive(obj, (newaxis.name,), interp1d)
    return result.transpose(obj.dims) # transpose back to original dimensions

interp1d = interp1d_numpy # alias

def interp2d_mpl(obj, newaxes, axes=None, order=1):
    """ bilinear interpolation: wrapper around mpl_toolkits.basemap.interp

    input:
	obj : Dimarray
	newaxes: list of Axis object, or list of 1d arrays
	axes, optional: list of str (axis names), required if newaxes is a list of arrays

    output:
	interpolated data (n-d)
    """
    from mpl_toolkits.basemap import interp

    # make sure input axes have the valid format
    newaxes = Axes.from_list(newaxes, axes) # valid format
    newaxes.sort(obj.dims) # re-order according to object's dimensions
    x, y = newaxes  # 2-d interpolation

    # make new grid 2-D
    x1, x1 = np.meshgrid(x.values, y.values, indexing='ij')

    def interp2d(obj, order=1):
	""" 2-D interpolation function appled recursively on the object
	"""
	x0, y0 = obj.axes[x.name].values, obj.axes[y.name].values
	res = interp(obj.values, x0, y0, x1, y1, order=order)
	return obj._constructor(res, newaxes, **obj._metadata)

    result = apply_recursive(obj, (x.name, y.name), interp2d)
    return result.transpose(obj.dims) # transpose back to original dimensions

interp2d = interp2d_mpl # alias
