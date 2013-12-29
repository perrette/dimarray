""" Dimarray methods to deal with axis alignment and interpolation

NOTE: these functions are used as methods for Dimarray, thus the first argument 
`self`
"""
import numpy as np
from axes import Axis, Axes, GroupedAxis

def reindex_axis(self, newaxis, axis=0, method='index'):
    """ reindex an array along an axis

    Input:
	- newaxis: Axis or array-like new axis
	- axis   : axis number or name
	- method : "index", "nearest", "interp" (see xs)

    Output:
	- Dimarray
    """
    if isinstance(newaxis, Axis):
	newaxis = newaxis.values
	axis = newaxis.name

    axis_id = self.axes.get_idx(axis)
    axis_nm = self.axes.get_idx(axis)
    ax = self.axes[axis_id] # Axis object

    # do nothing if axis is same or only None element
    if ax.values[0] is None or np.all(newaxis==ax.values):
	return self

    # indices along which to sample
    if method in ("nearest","index",None):

	indices = np.empty(np.size(newaxis), dtype=int)
	indices.fill(-1)

	# locate elements one by one...
	for i, val in enumerate(newaxis):
	    try:
		idx = ax.loc.locate(val, method=method)
		indices[i] = idx

	    except Exception, msg:
		# not found, will be filled with Nans
		pass

	# prepare array to slice
	values = self.values.take(indices, axis=axis_id)

	# missing indices
	# convert int to floats if necessary
	if values.dtype == np.dtype(int):
	    values = np.array(values, dtype=float, copy=False)

	missing = (slice(None),)*axis_id + np.where(indices==-1)
	#missing = ndindex(np.where(indices==-1), axis_id)
	values[missing] = np.nan # set missing values to NaN

    elif method == "interp":
	raise NotImplementedError(method)

    else:
	#return self.interp1d(newaxis, axis)
	raise ValueError(method)

    # new Dimarray
    # ...replace the axis
    new_ax = Axis(newaxis, ax.name)
    axes = self.axes.copy()
    axes[axis_id] = new_ax # replace the new axis

    # ...initialize Dimarray
    obj = self._constructor(values, axes, **self._metadata)
    return obj


def _reindex_axes(self, axes, method=None):
    """ reindex according to a list of axes
    """
    obj = self
    newdims = [ax2.name for ax2 in axes]
    for ax in self.axes:
	if ax.name in newdims:
	    newaxis = axes[ax.name].values
	    obj = obj.reindex_axis(newaxis, axis=ax.name, method=method)

    return obj

def reindex_like(self, other, method=None):
    """ reindex like another axis

    note: only reindex axes which are present in other
    """
    return self._reindex_axes(other.axes, method=method)
