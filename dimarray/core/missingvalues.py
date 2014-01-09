""" methods to deal with missing values
"""
import numpy as np

def isnan(a):
    """ analogous to numpy's isnan
    """
    return a._constructor(np.isnan(a.values), a.axes, **a._metadata)

def dropna(a, axis=None, how='any', thresh=None):
    """ analogous to pandas' dropna

    Examples:
    ---------

    1-Dimension
    >>> a = GeoArray([1.,2,3],time=[1950, 1955, 1960])
    >>> a.ix[1] = np.nan
    >>> a
    geoarray: 2 non-null elements (1 null)
    dimensions: 'time'
    0 / time (3): 1950 to 1960
    array([  1.,  nan,   3.])
    >>> a.dropna()
    geoarray: 2 non-null elements (0 null)
    dimensions: 'time'
    0 / time (2): 1950 to 1960
    array([ 1.,  3.])

    Multi-dimensional
    >>> a = GeoArray(np.arange(3*2)).reshape((3,2))

    """
    # special case: flatten array
    if axis is None:
	valid = ~np.isnan(a.values)
	return a[valid]

    idx, name = a._get_axis_info(axis)

    nans = ~isnan(a) # keep dims
    count_nans_axis = nans.sum(axis=[dim for dim in a.dims if dim != name]) # number of points valid along that axis

    # pick up only points whose number of nans is below the threshold
    if thresh is None: 
	thresh = 1

    indices = count_nans_axis < thresh

    return a.take(indices, axis=idx)
