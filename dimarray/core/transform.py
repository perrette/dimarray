""" Numpy-like Axis transformations
"""
import numpy as np
from functools import partial, wraps

from dimarray.decorators import format_doc
from dimarray.tools import anynan, is_DimArray

from axes import Axis, Axes, GroupedAxis

# check whether bottleneck is present
try:
    import bottleneck
    _hasbottleneck = True

except ImportError:
    _hasbottleneck = False

#
# Some general documention which is used in several methods
#

__all__ = []

_doc_axis = """
axis : int or str or tuple
      axis along which to apply the tranform. 
      Can be given as axis position (`int`), as axis name (`str`), as a 
      `list` or `tuple` of axes (positions or names) to collapse into one axis before 
      applying transform. If `axis` is `None`, just apply the transform on
      the flattened array consistently with numpy (in this case will return
      a scalar).
      Default is `{default_axis}`.
""".strip()

_doc_skipna = """
skipna : bool
    If True, treat NaN as missing values (either using MaskedArray or,
        when available, specific numpy function)
""".strip()

_doc_numpy = """ Analogous to numpy's {func}

{func}(..., axis=None, skipna=False, ...)

Accepts the same parameters as the equivalent numpy function, 
with modified behaviour of the `axis` parameter and an additional 
`skipna` parameter to handle NaNs (by default considered missing values)

Parameters
----------

{axis}
{skipna}

"..." stands for any other parameters required by the function, and depends
on the particular function being called 

Returns
-------
DimArray, or numpy array or scalar (e.g. in some cases if `axis` is None)

See help on numpy.{func} or numpy.ma.{func} for other parameters 
and more information.

See Also
--------
apply_along_axis: is called by this method
to_MaskedArray: is used if skipna is True
""".format(axis=_doc_axis, skipna=_doc_skipna, func="{func}")

#
# Actual transforms
#

@format_doc(skipna=_doc_skipna, axis=_doc_axis.format(default_axis="None"))
def apply_along_axis(self, func, axis=None, skipna=False, args=(), **kwargs):
    """ Apply along-axis numpy method to DimArray

    apply_along_axis(self, ...)
    Where ... are the parameters below:

    Parameters
    ----------
    func : numpy function name (`str`)
    {axis}
    {skipna}
    args : variable list of arguments before "axis"
    kwargs : variable dict of keyword arguments after "axis"
    
    Returns
    -------
    DimArray, or scalar 

    Examples
    --------
    >>> import dimarray as da
    >>> a = da.DimArray([[0,1],[2,3.]])
    >>> b = a.copy()
    >>> b[0,0] = np.nan
    >>> c = da.stack([a,b],keys=['a','b'],axis='items')
    >>> c
    dimarray: 7 non-null elements (1 null)
    0 / items (2): a to b
    1 / x0 (2): 0 to 1
    2 / x1 (2): 0 to 1
    array([[[  0.,   1.],
            [  2.,   3.]],
    <BLANKLINE>
           [[ nan,   1.],
            [  2.,   3.]]])
    >>> c.sum(axis=0)
    dimarray: 3 non-null elements (1 null)
    0 / x0 (2): 0 to 1
    1 / x1 (2): 0 to 1
    array([[ nan,   2.],
           [  4.,   6.]])
    >>> c.sum(0, skipna=True)
    dimarray: 4 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (2): 0 to 1
    array([[ 0.,  2.],
           [ 4.,  6.]])
    >>> c.median(0)
    dimarray: 3 non-null elements (1 null)
    0 / x0 (2): 0 to 1
    1 / x1 (2): 0 to 1
    array([[ nan,   1.],
           [  2.,   3.]])
    """


    # Deal with `axis` parameter, whether `int`, `str` or `tuple`
    obj, idx, name = _deal_with_axis(self, axis)

    # if function is provided by name, determine the proper function to use 
    if type(func) is str:
        funcname = func
        try:
            func = _get_func(funcname, skipna)
        except (AttributeError, AssertionError) as msg:
            #msg = funcname+" was not found in bottleneck, numpy, numpy.ma"
            raise ValueError(msg)

    # call function
    kwargs['axis'] = idx # only pass axis, not skipna
    result = func(obj.values, *args, **kwargs)
    funcname = func.__name__

    # If `axis` was None (operations on the flattened array), just returns the numpy array
    if axis is None or not isinstance(result, np.ndarray):
        return result

    #
    # New axes
    #
    # standard case: collapsed axis
    if np.ndim(result) == obj.ndim - 1:
        newaxes = [ax for ax in obj.axes if ax.name != name]

    # cumulative functions: axes remain unchanged
    elif funcname in ('cumsum','cumprod','gradient'):
        newaxes = obj.axes.copy() 

    # diff: reduce axis size by one
    elif funcname == "diff":
        oldaxis = obj.axes[idx]
        newaxis = oldaxis[1:]  # assume backward differencing 
        newaxes = obj.axes.copy() 
        newaxes[idx] = newaxis

    # axes do not fit for some reason
    else:
        raise Exception("cannot find new axes for this transformation: "+repr(funcname))

    newobj = obj._constructor(result, newaxes, **obj._metadata)

    # add stamp
    #stamp = "{transform}({axis})".format(transform=funcname, axis=str(obj.axes[idx]))
    #newobj._metadata_stamp(stamp)

    return newobj

class _MaskedArrayFunc(object):
    """ switcher between numpy and numpy.ma functions if skipna is True
    depends on whether nans are present in an array
    """
    def __init__(self, name):
        assert hasattr(np.ma, name) and hasattr(np, name), "function not present in numpy or numpy.ma "+name
        self.__name__ = name  # function name

    def __call__(self, values, *args, **kwargs):

        # transform numpy array to masked array if needed
        if anynan(values):
            values = np.ma.array(values, mask=np.isnan(values))
            func = getattr(np.ma, self.__name__)

        else:
            func = getattr(np, self.__name__)

        result = func(values, *args, **kwargs) 

        # transform back to numpy array
        if np.ma.isMaskedArray(result):
            result = result.filled(np.nan)

        return result

def _median_with_nan(values, *args, **kwargs):
    """ replace "median" if skipna is False
    
    numpy's median ignore NaNs as long as less than 50% 
    modify this behaviour and return NaN just as any other operation would
    """
    if _hasbottleneck:
        result = bottleneck.median(values, *args, **kwargs)
    else:
        result = np.median(values, *args, **kwargs)

    if anynan(values):
        if np.size(result) == 1: 
            result = np.nan
        else:
            axis = kwargs.pop('axis', None)
            nans = anynan(values, axis=axis) # determine where the nans should be
            result[nans] = np.nan

    return result

def _get_func(funcname, skipna):
    """ return a function based on its name and whether or not nans should be skipped
    """
    # At the time of writing this module, bottleneck functions were:
    # with nan as prefix : sum,max,min,argmin,argmax,mean,median,rankdata,std,var
    # and median, 
    # and move_mean,move_median,move_max,move_min,move_std,move_sum
    # and the same as move_nan<>, except that move_nanmedian was not present

    assert not funcname.startswith('nan'), "enter function name without nan, or provide the actual function as argument"

    #
    # ignore NaNs
    #
    if skipna:

        # check if present in bottleneck
        if _hasbottleneck:
            if funcname.startswith('move_'):
                funcname2 = 'move_nan'+funcname[6:]

            else:
                funcname2 = "nan"+funcname

            if hasattr(bottleneck, funcname2):
                return getattr(bottleneck, funcname2)

        # now try a 'nan...' function from numpy 
        funcname2 = "nan"+funcname
        if hasattr(np, funcname2):
            return getattr(np, funcname2)

        # function not present in bottleneck: by default switch back to numpy/numpy.ma
        return _MaskedArrayFunc(funcname)

    #
    # do not ignore nans
    #
    assert not skipna

    # modify default median behaviour to put nans in the result
    if funcname == 'median':
        return _median_with_nan

    # use bottleneck if existing
    if _hasbottleneck and hasattr(bottleneck, funcname):
        return getattr(bottleneck, funcname)

    # otherwise basic numpy function
    return getattr(np, funcname)

def _deal_with_axis(obj, axis):
    """ deal with the `axis` parameter 

    Parameters
    ----------
        obj: DimArray object
        axis: `int` or `str` or `tuple` or None

    Returns
    -------
        newobj: reshaped obj if axis is tuple otherwise obj
        idx   : axis index
        name  : axis name
    """
    # before applying the function on the collapsed array
    if type(axis) in (tuple, list):
        idx = 0
        newobj = obj.group(axis, insert=idx)
        #idx = obj.dims.index(axis[0])  # position where the new axis has been inserted
        #ax = newobj.axes[idx] 
        ax = newobj.axes[0]
        name = ax.name

    else:
        newobj = obj
        idx, name = obj._get_axis_info(axis) 

    return newobj, idx, name

#
# Technicalities to apply numpy methods
#
class _NumpyDesc(object):
    """ to apply a numpy function which reduces an axis
    """
    def __init__(self, numpy_method):
        """
        numpy_method: as a string name of the numpy function to apply
        """
        assert type(numpy_method) is str, "can only provide method name as a string"
        self.numpy_method = numpy_method

    def __get__(self, obj, cls=None):
        """
        """
        # convert self.apply to an actual function
        #newmethod = wraps(apply_along_axis)(partial(apply_along_axis, obj, self.numpy_method))
        newmethod = partial(apply_along_axis, obj, self.numpy_method)

        # Update doc string
        newmethod.__doc__ = _doc_numpy.format(func=self.numpy_method, default_axis=None)

        # update the name 
        newmethod.__name__ = self.numpy_method

        return newmethod

median = _NumpyDesc("median")

# basic, unmodified transforms

prod = _NumpyDesc("prod")
sum  = _NumpyDesc("sum")

# here just for checking
_mean = _NumpyDesc("mean")
_std = _NumpyDesc("std")
_var = _NumpyDesc("var")

min = _NumpyDesc("min")
max = _NumpyDesc("max")

ptp = _NumpyDesc("ptp")
all = _NumpyDesc("all")
any = _NumpyDesc("any")

#cumsum = _NumpyDesc("cumsum", axis=-1)
#cumprod = _NumpyDesc("cumprod", axis=-1)

def cumsum(a, axis=-1, skipna=False):
    return apply_along_axis(a, 'cumsum', axis=axis, skipna=skipna)

def cumprod(a, axis=-1, skipna=False):
    return apply_along_axis(a, 'cumprod', axis=axis, skipna=skipna)

#
# Special behaviour for argmin and argmax: return axis values instead of integer position
#
@format_doc(default_axis="None")
@format_doc(axis=_doc_axis, skipna=_doc_skipna)
def argmin(self, axis=None, skipna=False):
    """ similar to numpy's argmin, but return axis values instead of integer position

    Parameters
    ----------
    {axis}
    {skipna}
    """
    obj, idx, name = _deal_with_axis(self, axis)
    res = apply_along_axis(obj, 'argmin', axis=idx, skipna=skipna)

    # along axis: single axis value
    if axis is not None: # res is DimArray
        res.values = obj.axes[idx].values[res.values] 
        return res

    # flattened array: tuple of axis values
    else: # res is ndarray
        res = np.unravel_index(res, obj.shape)
        return tuple(obj.axes[i].values[v] for i, v in enumerate(res))

@format_doc(default_axis="None")
@format_doc(axis=_doc_axis, skipna=_doc_skipna)
def argmax(self, axis=None, skipna=False):
    """ similar to numpy's argmax, but return axis values instead of integer position

    Parameters
    ----------
    {axis}
    {skipna}
    """
    obj, idx, name = _deal_with_axis(self, axis)
    res = apply_along_axis(obj, 'argmax', axis=idx, skipna=skipna)

    # along axis: single axis value
    if axis is not None: # res is DimArray
        res.values = obj.axes[idx].values[res.values] 
        return res

    # flattened array: tuple of axis values
    else: # res is ndarray
        res = np.unravel_index(res, obj.shape)
        return tuple(obj.axes[i].values[v] for i, v in enumerate(res))

#
# Also define numpy.diff as method, with additional options
#

@format_doc(default_axis=-1)
@format_doc(axis=_doc_axis)
def diff(self, axis=-1, scheme="backward", keepaxis=False, n=1):
    """ Analogous to numpy's diff

    Calculate the n-th order discrete difference along given axis.

    The first order difference is given by ``out[n] = a[n+1] - a[n]`` along
    the given axis, higher order differences are calculated by using `diff`
    recursively.

    Parameters
    ----------
    {axis}

    scheme : str, optional
        determines the values of the resulting axis
        - "forward" : diff[i] = x[i+1] - x[i]
        - "backward": diff[i] = x[i] - x[i-1]
        - "centered": diff[i] = x[i+1/2] - x[i-1/2]
        Default is "backward"

    keepaxis : bool, optional
            if True, keep the initial axis by padding with NaNs
            Only compatible with "forward" or "backward" differences
            Default is False

    n : int, optional
        The number of times values are differenced.
        Default is one

    Returns
    -------
    diff : DimArray
        The `n` order differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`.

    Examples
    --------

    Create some example data

    >>> import dimarray as da
    >>> v = da.DimArray([1,2,3,4], ('time', np.arange(1950,1954)), dtype=float)
    >>> s = v.cumsum()
    >>> s 
    dimarray: 4 non-null elements (0 null)
    0 / time (4): 1950 to 1953
    array([  1.,   3.,   6.,  10.])

    `diff` reduces axis size by one, by default

    >>> s.diff()
    dimarray: 3 non-null elements (0 null)
    0 / time (3): 1951 to 1953
    array([ 2.,  3.,  4.])

    The `keepaxis=` parameter fills array with `nan` where necessary to keep the axis unchanged. Default is backward differencing: `diff[i] = v[i] - v[i-1]`.

    >>> s.diff(keepaxis=True)
    dimarray: 3 non-null elements (1 null)
    0 / time (4): 1950 to 1953
    array([ nan,   2.,   3.,   4.])

    But other schemes are available to control how the new axis is defined: `backward` (default), `forward` and even `centered`

    >>> s.diff(keepaxis=True, scheme="forward") # diff[i] = v[i+1] - v[i]
    dimarray: 3 non-null elements (1 null)
    0 / time (4): 1950 to 1953
    array([  2.,   3.,   4.,  nan])

    The `keepaxis=True` option is invalid with the `centered` scheme, since every axis value is modified by definition:

    >>> s.diff(axis='time', scheme='centered')
    dimarray: 3 non-null elements (0 null)
    0 / time (3): 1950.5 to 1952.5
    array([ 2.,  3.,  4.])
    """
    # If `axis` is None (operations on the flattened array), just returns the numpy array
    if axis is None:
        return np.diff(self.values, n=n, axis=None)

    # Deal with `axis` parameter, whether `int`, `str` or `tuple`
    # possibly grouping dimensions if axis is tuple
    obj, idx, name = _deal_with_axis(self, axis)

    # Recursive call if n > 1
    if n > 1:
        obj = obj.diff(n=n-1, axis=idx, scheme=scheme, keepaxis=keepaxis)
        n = 1

    # n = 1
    assert n == 1, "n must be integer greater or equal to one"

    # Compute differences
    result = np.diff(obj.values, axis=idx)

    # Old axis along diff
    oldaxis = obj.axes[idx]

    # forward differencing
    if scheme == "forward":

        # keep axis: pad last element with NaNs
        if keepaxis:
            result = _append_nans(result, axis=idx)
            newaxis = oldaxis.copy()

        # otherwise just shorten the axis
        else:
            newaxis = oldaxis[:-1]

    elif scheme == "backward":

        # keep axis: pad first element with NaNs
        if keepaxis:
            result = _append_nans(result, axis=idx, first=True)
            newaxis = oldaxis.copy()

        # otherwise just shorten the axis
        else:
            newaxis = oldaxis[1:]

    elif scheme == "centered":

        # keep axis: central difference + forward/backward diff at the edges
        if keepaxis:
            #indices = range(oldaxis.size)
            raise ValueError("keepaxis=True is not compatible with centered differences")
            #central = obj.values.take(indices[2:], axis=idx) \
            #        -  obj.values.take(indices[:-2], axis=idx)
            #start = obj.values.take([1], axis=idx) \
            #        -  obj.values.take([0], axis=idx)
            #end = obj.values.take([-1], axis=idx) \
            #        -  obj.values.take([-2], axis=idx)
            #result = np.concatenate((start, central, end), axis=idx)
            #newaxis = oldaxis.copy()

        else:
            axisvalues = 0.5*(oldaxis.values[:-1]+oldaxis.values[1:])
            newaxis = Axis(axisvalues, name)

    else:
        raise ValueError("scheme must be one of 'forward', 'backward', 'central', got {}".format(scheme))

    newaxes = obj.axes.copy() 
    newaxes[idx] = newaxis
    newobj = obj._constructor(result, newaxes, **obj._metadata)

    return newobj

def _append_nans(result, axis, first=False):
    """ insert a slice of NaNs at the front of an array along axis
    or append the slice if append is True

    axis: `int`
    """
    nan_slice = np.empty_like(result.take([0], axis=axis)) # make a slice ...
    nan_slice.fill(np.nan) # ...filled with NaNs

    # Insert as first element
    if first:
        result = np.concatenate((nan_slice, result), axis=axis)

    # Append
    else:
        result = np.concatenate((result, nan_slice), axis=axis)

    return result

#def _apply_minmax(obj, funcname, axis=None, skipna=False, args=(), **kwargs):
#    """ apply min/max/argmin/argmax
#
#    special behaviour for these functions, with `keepdims` parameter
#    """
#    # get actual axis values instead of numpy's integer index
#    if funcname in ("argmax","argmin","nanargmax","nanargmin"):
#        assert axis is not None, "axis must not be None for "+funcname+", or apply on values"
#        return obj.axes[idx].values[result] 



#
# Define weighted mean/std/var
#
_doc_weights = dict(

        axis = """
    axis : int or str or tuple, optional
        axis or group of axes to apply the transform on
        """.strip(),

        skipna = """
    skipna : bool, optional
        ignore missing values (nans) prior to transformation
        Default is False.""".strip(),

        weights = """
    weights : array-like or callable or dict, optional
        if provided, is used instead if individual axes' `weights` attributes

        A weights array can be built from individual axes' weights, either
        as a parameter to this function, or as a permanent axis attribute
        defined for the relevant axes. Weights can be of the form:

        - 1-D array-like : for 1-D arrays or if the transformation
        is to be applied to one axis only (via `axis` parameter)

        - callable : like above, will be applied on the axis specified 
        by the `axis` parameter, if provided

        If passed as a parameter to the weighted transformation, weights
        can also be provided as a dictionary. The keys must be axis names 
        or integer ranks, and values one of the accepted types.""".strip(),

        note = """
    Note
    ----
    The weights actually used in the transformation can be checked 
    via the DimArray._get_weights() method (experimental)""".strip(),

        descr = """
    This transformation can be weighted if a non-None `weights` parameter is provided, or if
    one of the axes has a non-None `weights` attribute. Otherwise, a standard, unweighted 
    transformation is performed.""".strip(),

)

#
# Return a weight for each array element, used for `mean`, `std` and `var`
#
@format_doc(**_doc_weights)
def _get_weights(self, axis=None, mirror_nans=True, weights=None):
    """ Build array of weights based on individual axes' weights attribute

    Parameters
    ----------
    axis : int or str, optional
        if None a N-D weight is created, otherwise 1-D only
    mirror_nans : bool, optional
        mirror `values`'s NaNs in created weights (True by default)
    {weights}

    Returns
    -------
    full_weights : DimArray of weights, or None

    See Also
    --------
    DimArray.mean, DimArray.var, DimArray.std
    """
    # avoid circular import
    from dimarray import DimArray

    if axis is None:
        dims = self.dims

    elif type(axis) is tuple:
        dims = axis

    else:
        dims = (axis,)

    # for 1-D array, same as axis is None is like axis=0
    if axis is None and self.ndim == 1:
        axis = 0

    # check weights type
    array_type = (isinstance(weights, np.ndarray) or isinstance(weights, DimArray)) and weights.ndim == 1
    dict_type = isinstance(weights, dict)
    assert weights is None or callable(weights) or array_type or dict_type, "unexpected type for `weights`"
    assert weights is None or not (not dict_type and axis is None), "`weights` type incompatible with flattening operation, provide `axis` or use dict-type weights"
    assert weights is None or not (not dict_type and type(axis) is tuple), "`weights` type incompatible with flattening operation, use dict-type weights"

    # make weight dict-like to be applicable (operation on single axis)
    if not dict_type: 
        weights = {axis: weights}

    # axes over which the weight is defined
    axes = [ax for ax in self.axes if ax.name in dims]

    # Create weights from the axes 
    full_weights = 1.
    for axis in dims:
        ax = self.axes[axis]

        # get proper weights array: from `Axis` or user-provided on-the-fly
        if weights is not None and axis in weights.keys():
            axis_weights = weights[axis]
        else:
            axis_weights = ax.weights

        # if weight is None, just skip it
        if axis_weights is None: 
            continue

        # broadcast and multiply weights for all axes involved 
        full_weights = ax._get_weights(axis_weights).reshape(dims) * full_weights

    if full_weights is 1.:
        return None
    else:
        full_weights = full_weights.values

    # Now create a dimarray and make it full size
    # ...already full-size
    if full_weights.shape == self.shape:
        full_weights = DimArray(full_weights, self.axes)

    # ...need to be expanded
    elif full_weights.shape == tuple(ax.size for ax in axes):
        full_weights = DimArray(full_weights, axes).expand(self.dims)

    else:
        try:
            full_weights = full_weights + np.zeros_like(self.values)

        except:
            raise ValueError("weight dimensions are not conform")
        full_weights = DimArray(full_weights, self.axes)

    # add NaNs in when necessary
    if mirror_nans and anynan(self.values):
        full_weights.values[np.isnan(self.values)] = np.nan
    
    assert full_weights is not None
    return full_weights


@format_doc(**_doc_weights)
def mean(self, axis=None, skipna=False, weights=None):
    """ mean over an axis or group of axes, possibly weighted 

    {descr} 

    Parameters
    ----------
    {axis}
    {skipna}
    {weights}
    
    Returns
    -------
    DimArray instance or scalar, consistently with ndarray behaviour

    {note}

    See Also
    --------
    DimArray.var, DimArray.std, DimArray._get_weights

    Example
    -------
    >>> from dimarray import DimArray
    >>> np.random.seed(0) # to make results reproducible
    >>> v = DimArray(np.random.rand(3,2), axes=[[-80, 0, 80], [-180, 180]], dims=['lat','lon'])

    Classical, unweighted mean:

    >>> v.mean() 
    0.58019972362897432

    Weighted mean

    >>> w = np.cos(np.radians(v.lat))
    >>> v.mean(weights={{'lat':w}})  # only lat axis is weighted
    0.57628879031663871

    Make the change permanent

    >>> v.axes['lat'].weights = w
    >>> v.mean()
    0.57628879031663871

    Check the weights being used (experimental)

    >>> v._get_weights()
    dimarray: 6 non-null elements (0 null)
    0 / lat (3): -80 to 80
    1 / lon (2): -180 to 180
    array([[ 0.17364818,  0.17364818],
           [ 1.        ,  1.        ],
           [ 0.17364818,  0.17364818]])
    """

    # Proceed to a weighted mean
    weights = self._get_weights(weights=weights, axis=axis, mirror_nans=skipna)

    # if no weights, just apply numpy's mean
    if weights is None:
        return apply_along_axis(self, "mean", axis=axis, skipna=skipna)

    # weighted mean
    sum_values = apply_along_axis(self*weights, "sum", axis=axis, skipna=skipna)
    sum_weights = apply_along_axis(weights, "sum", axis=axis, skipna=skipna)
    return sum_values / (sum_weights + 0.)

@format_doc(**_doc_weights)
def var(self, axis=None, skipna=False, weights=None, ddof=0):
    """ variance over an axis or group of axes, possibly weighted 

    Parameters
    ----------
    {axis}
    {skipna}
    {weights}
    ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
             ``N - ddof``, where ``N`` represents the number of elements. By default `ddof` is zero.
            Note ddof is ignored when weights are used
            Default is 0.
    
    Returns
    -------
    DimArray instance or scalar, consistently with ndarray behaviour

    {note}

    See Also
    --------
    DimArray.mean, DimArray.std, DimArray._get_weights
    """
    # Proceed to a weighted var
    weights = self._get_weights(weights=weights, axis=axis, mirror_nans=skipna)

    # if no weights, just apply numpy's var
    if weights is None:
        return apply_along_axis(self, "var", axis=axis, skipna=skipna, ddof=ddof)

    # weighted mean
    mean = self.mean(axis=axis, skipna=skipna, weights=weights)
    dev = (self-mean)**2
    return dev.mean(axis=axis, skipna=skipna, weights=weights)

# add standard deviation
def std(self, *args, **kwargs):
    return self.var(*args, **kwargs)**0.5

std.__doc__ = var.__doc__.replace('variance','standard deviation').replace('var','std').replace('DimArray.std','DimArray.var') # last one for See Also
