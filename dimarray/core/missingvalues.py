""" methods to deal with missing values
"""
import numpy as np
from tools import is_DimArray

def _isnan(a, na=np.nan):
    """ analogous to numpy's isnan
    """
    if np.isnan(na):
        return np.isnan(a)
    else:
        return a== na

def is_boolean_array(value):
    """ 
    >>> from dimarray import DimArray
    >>> a = DimArray([1,2,3])
    >>> is_boolean_array(a)
    False
    >>> is_boolean_array(a>1)
    True
    """
    return (isinstance(value, np.ndarray) or is_DimArray(value)) \
            and value.dtype is np.dtype('bool')

def _matches(a, value):

    if is_boolean_array(value):
        # boolean array accepted
        test = np.asarray(value)

    elif np.iterable(value):
        test = np.any([_matches(a, val) for val in value], axis=0)

    else:
        test = a == value
    return test

def isnan(a, na=np.nan):
    """ analogous to numpy's isnan
    """
    return a._constructor(_isnan(a, na=na), a.axes, **a._metadata)

def setna(a, value, na=np.nan, inplace=False):
    """ set a value as missing

    Parameters
    ----------
    value: the values to set to na
    na: the replacement value (default np.nan)
    >>> from dimarray import DimArray
    >>> a = DimArray([1,2,-99])
    >>> a.setna(-99)
    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  1.,   2.,  nan])
    >>> a.setna([-99, 2]) # sequence
    dimarray: 1 non-null elements (2 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  1.,  nan,  nan])
    >>> a.setna(a > 1) # boolean
    dimarray: 2 non-null elements (1 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  1.,  nan, -99.])
    >>> a = DimArray([[1,2,-99]])  # multi-dim
    >>> a.setna([-99, a>1])  # boolean
    dimarray: 1 non-null elements (2 null)
    dimensions: 'x0', 'x1'
    0 / x0 (1): 0 to 0
    1 / x1 (3): 0 to 2
    array([[  1.,  nan,  nan]])
    """
    return a.put(na, _matches(a.values, value), convert=True, inplace=inplace)

def fillna(a, value, inplace=False, na=np.nan):
    """
    >>> from dimarray import DimArray
    >>> a = DimArray([1,2,np.nan])
    >>> a.fillna(-99)
    dimarray: 3 non-null elements (0 null)
    dimensions: 'x0'
    0 / x0 (3): 0 to 2
    array([  1.,   2., -99.])
    """
    return a.put(value, _isnan(a.values, na=na), convert=True, inplace=inplace)

def dropna(a, axis=0, minvalid=None, na=np.nan):
    """ drop nans along an axis

    Parameters
    ----------
    axis: axis position or name or list of names
    minvalid, optional: min number of valid point in each slice along axis values
        by default all the points

    Examples
    --------

    1-Dimension
    >>> from dimarray import DimArray
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
    >>> a = DimArray([[ np.nan, 2., 3.],[ np.nan, 5., np.nan]])
    >>> a
    dimarray: 3 non-null elements (3 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (3): 0 to 2
    array([[ nan,   2.,   3.],
           [ nan,   5.,  nan]])
    >>> a.dropna(axis=1)
    dimarray: 2 non-null elements (0 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (1): 1 to 1
    array([[ 2.],
           [ 5.]])
    >>> a.dropna(axis=1, minvalid=1)  # minimum number of valid values, equivalent to `how="all"` in pandas
    dimarray: 3 non-null elements (1 null)
    dimensions: 'x0', 'x1'
    0 / x0 (2): 0 to 1
    1 / x1 (2): 1 to 2
    array([[  2.,   3.],
           [  5.,  nan]])
    """
    assert axis is not None, "axis cannot be None for dropna"
#    # if None, all axes
#    if axis is None:
#        for dim in a.dims:
#            a = dropna(a, axis=dim, minvalid=minvalid)
#        return a

    idx, name = a._get_axis_info(axis)

    if a.ndim == 1:
        return  a[~_isnan(a.values, na=na)]

    else:
        nans = isnan(a, na=na) 
        nans = nans.group([dim for dim in a.dims if dim != name], insert=0) # in first position
        count_nans_axis = nans.sum(axis=0) # number of points valid along that axis
        count_vals_axis = (~nans).sum(axis=0) # number of points valid along that axis
        #count_nans_axis = nans.sum(axis=[dim for dim in a.dims if dim != name]) # number of points valid along that axis
        #count_vals_axis = nans.sum(axis=[dim for dim in a.dims if dim != name]) # number of points valid along that axis

    # pick up only points whose number of nans is below the threshold
    if minvalid is None: 
        maxna = 0
    else:
        maxna = nans.axes[0].size - minvalid

    indices = count_nans_axis <= maxna

    return a.take(indices, axis=idx)
