""" methods to deal with missing values
"""
import numpy as np

def _isnan(a, na=np.nan):
    """ analogous to numpy's isnan
    """
    if np.isnan(na):
	return np.isnan(a)
    else:
	return a== na

def isnan(a, na=np.nan):
    """ analogous to numpy's isnan
    """
    return a._constructor(_isnan(a, na=na), a.axes, **a._metadata)

def setna(a, value, na=np.nan, inplace=False):
    """ set a value as missing

    value: the values to set to na
    na: the replacement value (default np.nan)

    >>> a = DimArray([1,2,-99])
    >>> a.setna(-99)
    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  1.,   2.,  nan])
    """
    return a.put(na, a.values==value, convert=True, inplace=inplace)

def fillna(a, value, inplace=False, na=np.nan):
    """
    >>> a = DimArray([1,2,np.nan])
    >>> a.fillna(-99)
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  1.,   2., -99.])
    """
    return a.put(value, _isnan(a.values, na=na), convert=True, inplace=inplace)

def dropna(a, axis=0, minval=None, na=np.nan):
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
    >>> a = DimArray(np.arange(3*2).reshape((3,2)))+0.
    >>> a.ix[0, 1] = np.nan
    >>> a 
    dimarray: 5 non-null elements (1 null)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (2): 0 to 1
    array([[  0.,  nan],
           [  2.,   3.],
           [  4.,   5.]])
    >>> a.dropna() # default axis=0
    dimarray: 4 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 1 to 2
    1 / x1 (2): 0 to 1
    array([[ 2.,  3.],
           [ 4.,  5.]])
    >>> a.dropna(axis=1)
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (3): 0 to 2
    1 / x1 (1): 0 to 0
    array([[ 0.],
           [ 2.],
           [ 4.]])
    """
    assert axis is not None, "axis cannot be None for dropna"
#    # if None, all axes
#    if axis is None:
#	for dim in a.dims:
#	    a = dropna(a, axis=dim, minval=minval)
#	return a

    idx, name = a._get_axis_info(axis)

    if a.ndim == 1:
	return  a[~_isnan(a.values, na=na)]

    else:
	nans = isnan(a, na=na) 
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
