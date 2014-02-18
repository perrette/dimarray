""" Statistical function Library (additionally to bounds methods such as mean, var etc...)
"""
import numpy as np
import dimarray as da

def percentile(a, pct, axis=0, name=None, out=None, overwrite_input=False):
    """ calculate percentile along an axis

    pct: float, percentile or sequence of percentiles (0< <100)
    axis, optional, default 0: axis along which to compute percentiles
    name, optional: name of the new percentile axis, if more than one pct. 
	By default, append "_pct" to the axis name on which the transformation
	is applied.

    out, overwrite_input: passed to numpy's percentile method (see documentation)

    Examples:
    ---------
    >>> np.random.seed(0) # for reproductibility of results
    >>> a = DimArray(np.random.randn(1000), dims=['sample'])
    >>> percentile(a, 50)
    -0.058028034799627745

    >>> percentile(a, [50, 95])
    dimarray: 2 non-null elements (0 null)
    dimensions: 'sample_percentile'
    0 / sample_percentile (2): 50 to 95
    array([-0.05802803,  1.66012041])
    """
    if not isinstance(a, da.DimArray):
	raise TypeError("Expected DimArray instance got {} of type {}".format(a, type(a)))
    pos, nm = a._get_axis_info(axis)
    results = np.percentile(a.values, pct, axis=pos, out=out, overwrite_input=overwrite_input)

    # If the result is scalar (pct is scalar and ), just return it
    if np.isscalar(results):
	return results

    # for scalar pct, results is a numpy array. Just reduce the axis.
    if np.isscalar(pct):
	results = da.DimArray(results, axes=[ax for ax in a.axes if ax.name != nm])

    # pct is array-like, recreate a Dimarray
    else:
	if name is None:
	    name = nm + '_percentile'
	results = da.from_arrays(results, keys=pct, axis=name)

    return results

def quantile(a, q, axis=0, name=None, out=None, overwrite_input=False):
    """ Same as percentile, but provide quantiles instead 
    
    q: quantile(s): must be between 0 and 1 instead of 0 and 100

    See help on percentile for full documentation.

    See Also:
    ---------
    percentile

    Example:
    -------
    >>> np.random.seed(0) # for reproductibility of results
    >>> a = DimArray(np.random.randn(1000), dims=['sample'])
    >>> quantile(a, [0.5, 0.95])
    dimarray: 2 non-null elements (0 null)
    dimensions: 'sample_quantile'
    0 / sample_quantile (2): 0.5 to 0.95
    array([-0.05802803,  1.66012041])
    """
    pos, nm = a._get_axis_info(axis)
    if name is None:
	name = nm + '_quantile'

    res = percentile(a, [qi*100 for qi in q], axis=axis, name=name, out=out, overwrite_input=overwrite_input)

    # change the percentile axis into quantile axis
    if not np.isscalar(q):
	res.axes[axis].values /= 100.

    return res
