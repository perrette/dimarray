""" methods to deal with missing values
"""
import numpy as np

def isnan(a):
    """ analogous to numpy's isnan
    """
    return a._constructor(np.isnan(a.values), a.axes, **a._metadata)

def dropna(a, axis=0, minval=None):
    """ drop nans along an axis

    axis: axis position or name or list of names
    minval, optional: min number of valid point in each slice along axis values
	by default all the points

    Examples:
    ---------

    1-Dimension
    >>> a = DimArray([1.,2,3],('time',[1950, 1955, 1960]))
    >>> a.ix[1] = np.nan
    >>> a
    dimarray: 2 non-null elements (1 null)
    dimensions: 'time'
    0 / time (3): 1950 to 1960
    array([  1.,  nan,   3.])
    >>> a.dropna()
    dimarray: 2 non-null elements (0 null)
    dimensions: 'time'
    0 / time (2): 1950 to 1960
    array([ 1.,  3.])

    Multi-dimensional
    TODO
  ##  >>> a = DimArray(np.arange(3*2).reshape((3,2)))
  ##  >>> a.ix[0, 1] = np.nan
  ##  >>> a
    """
    assert axis is not None, "axis cannot be None for dropna"
#    # if None, all axes
#    if axis is None:
#	for dim in a.dims:
#	    a = dropna(a, axis=dim, minval=minval)
#	return a

    idx, name = a._get_axis_info(axis)

    if a.ndim == 1:
	return  a[~isnan(a)]

    else:
	nans = isnan(a) 
	#nans = nans.group([dim for dim in a.dims if dim != name]) # in first position
	nans = nans.group([dim for dim in a.dims if dim != name]) # in first position
	count_nans_axis = nans.sum(axis=0) # number of points valid along that axis
	count_vals_axis = (~nans).sum(axis=0) # number of points valid along that axis
	#count_nans_axis = nans.sum(axis=[dim for dim in a.dims if dim != name]) # number of points valid along that axis
	#count_vals_axis = nans.sum(axis=[dim for dim in a.dims if dim != name]) # number of points valid along that axis

    # pick up only points whose number of nans is below the threshold
    if minval is None: 
	maxna = 1
    else:
	maxna = nans.axes[0].size - minval

    indices = count_nans_axis < maxna

    return a.take(indices, axis=idx)
