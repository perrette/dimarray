""" Numpy-like Axis transformations
"""
from functools import partial, wraps
import itertools
import numpy as np

# check whether bottleneck is present
try:
    import bottleneck
    _hasbottleneck = True
except ImportError:
    _hasbottleneck = False

from dimarray.tools import anynan, is_DimArray, format_doc
from dimarray.core.axes import Axis, Axes


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

    Notes
    -----
    If you have the bottleneck pacakge installed, its functions should work
    with `apply_along_axis`, such as move_mean

    Examples
    --------
    >>> import dimarray as da
    >>> a = da.DimArray([[0,1],[2,3.]])
    >>> b = a.copy()
    >>> b[0,0] = np.nan
    >>> c = da.stack([a,b],keys=['a','b'],axis='items')
    >>> c
    dimarray: 7 non-null elements (1 null)
    0 / items (2): 'a' to 'b'
    1 / x0 (2): 0 to 1
    2 / x1 (2): 0 to 1
    array([[[ 0.,  1.],
            [ 2.,  3.]],
    <BLANKLINE>
           [[nan,  1.],
            [ 2.,  3.]]])
    >>> c.sum(axis=0)
    dimarray: 3 non-null elements (1 null)
    0 / x0 (2): 0 to 1
    1 / x1 (2): 0 to 1
    array([[nan,  2.],
           [ 4.,  6.]])
    >>> c.sum(0, skipna=True)
    dimarray: 4 non-null elements (0 null)
    0 / x0 (2): 0 to 1
    1 / x1 (2): 0 to 1
    array([[0., 2.],
           [4., 6.]])
    >>> c.median(0)
    dimarray: 3 non-null elements (1 null)
    0 / x0 (2): 0 to 1
    1 / x1 (2): 0 to 1
    array([[nan,  1.],
           [ 2.,  3.]])
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
    # same for moving (rolling) operations.
    elif funcname in ('cumsum','cumprod','gradient') \
        or 'move_' in funcname or 'cum' in funcname:
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

    newobj = obj._constructor(result, newaxes, **obj.attrs)

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
    """Handle the `axis` parameter 

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
        newobj = obj.flatten(axis, insert=idx)
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
mean = _NumpyDesc("mean")
std = _NumpyDesc("std")
var = _NumpyDesc("var")

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
    array([ 1.,  3.,  6., 10.])

    `diff` reduces axis size by one, by default

    >>> s.diff()
    dimarray: 3 non-null elements (0 null)
    0 / time (3): 1951 to 1953
    array([2., 3., 4.])

    The `keepaxis=` parameter fills array with `nan` where necessary to keep the axis unchanged. Default is backward differencing: `diff[i] = v[i] - v[i-1]`.

    >>> s.diff(keepaxis=True)
    dimarray: 3 non-null elements (1 null)
    0 / time (4): 1950 to 1953
    array([nan,  2.,  3.,  4.])

    But other schemes are available to control how the new axis is defined: `backward` (default), `forward` and even `centered`

    >>> s.diff(keepaxis=True, scheme="forward") # diff[i] = v[i+1] - v[i]
    dimarray: 3 non-null elements (1 null)
    0 / time (4): 1950 to 1953
    array([ 2.,  3.,  4., nan])

    The `keepaxis=True` option is invalid with the `centered` scheme, since every axis value is modified by definition:

    >>> s.diff(axis='time', scheme='centered')
    dimarray: 3 non-null elements (0 null)
    0 / time (3): 1950.5 to 1952.5
    array([2., 3., 4.])
    """
    # If `axis` is None (operations on the flattened array), just returns the numpy array
    if axis is None:
        return np.diff(self.values, n=n, axis=None)

    # Deal with `axis` parameter, whether `int`, `str` or `tuple`
    # possibly flattening dimensions if axis is tuple
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

    newaxes = [ax.copy() if ax.name != name else newaxis for ax in obj.axes]
    newobj = obj._constructor(result, newaxes, **obj.attrs)

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
# linear interpolation along an axis
#

# sort the axis if needed, to apply numpy interp
def _interp_internal_maybe_sort(obj, axis, issorted):
    curaxis = obj.axes[axis]
    if issorted is None:
        issorted = np.all(curaxis.values[1:] >= curaxis.values[:-1])
    if not issorted:
        obj = obj.sort_axis(axis=axis)
    return obj

def _numpy_interp(x, xp, yp, left=None, right=None):
    """Compatibility function for numpy interp, since the behaviour changes for certain versions
    """
    y = np.interp(x, xp, yp, left=left, right=right)

    # numpy 1.10.1 messes with left...
    if np.__version__ == '1.10.1' and left is not None:
        imin = np.argmin(xp, axis=0)
        y[x == xp[imin]] = yp[imin]

    return y

# get interp weights
def _interp_internal_get_weights(oldx, newx):
    " compute necessary indices and weights to perform linear interpolation "
    newindices = _numpy_interp(newx, oldx, np.arange(oldx.size), left=-oldx.size, right=-1)
    left_idx = newindices == -oldx.size # out-of-bounds
    right_idx = newindices == -1
    lhs_idx = np.asarray(newindices, dtype=int)
    rhs_idx = np.asarray(np.ceil(newindices), dtype=int)
    frac = newindices - lhs_idx
    # return lhs_idx, rhs_idx, frac, left_idx, right_idx
    return {'lhs_idx':lhs_idx, 'rhs_idx':rhs_idx, 'frac':frac, 'left_idx':left_idx,'right_idx':right_idx}

# apply interp from weights
def _interp_internal_from_weight(arr, axis, left, right, lhs_idx, rhs_idx, frac, left_idx, right_idx):
    " numpy ==> numpy "
    # pre-broadcast dimensions
    if arr.ndim > 1:
        arr = arr.swapaxes(axis, 0) # make the interp axis the first axis
        _frac = frac[(slice(None),)+(None,)*(arr.ndim-1)] # broadcast frac for multiplication
    else:
        _frac = frac

    # compute the weighted sum
    vleft = arr[lhs_idx]
    vright = arr[rhs_idx]
    newval = vleft + _frac*(vright - vleft)

    # fill values
    newval[left_idx] = left
    newval[right_idx] = right

    # transpose back
    if arr.ndim > 1:
        newval = newval.swapaxes(axis, 0)
    return newval

def interp_axis(self, values, axis=0, left=np.nan, right=np.nan, issorted=None):
    """ interpolate along one axis

    Parameters
    ----------
    values : 1d array-like
    axis, optional : axis name or integer rank
        required unless values is an Axis
    left, right : fill_values at the edges
    issorted : None or bool, optional
        indicates wether the original axis is sorted, to skip pre-sorting step
        by default None: a check is performed, and the axis is sorted if needed

    Returns
    -------
    dima : interpolated DimArray 

    Examples
    --------
    >>> from dimarray import DimArray

    >>> a = DimArray([3,4], axes=[[1,3]])
    >>> a.interp_axis([1,2,3])
    dimarray: 3 non-null elements (0 null)
    0 / x0 (3): 1 to 3
    array([3. , 3.5, 4. ])
    
    Axis is not sorted

    >>> a = DimArray([3,0,1], axes=[[3,0,1]]) 
    >>> a.interp_axis([1,2,3])
    dimarray: 3 non-null elements (0 null)
    0 / x0 (3): 1 to 3
    array([1., 2., 3.])

    N-Dimensional

    >>> b = DimArray([[1,2,3],[4,5,6]], axes=[['a','b'], [0,1,2]]) # N-Dim
    >>> b.interp_axis([0.5, 1.5, 2], axis=1)
    dimarray: 6 non-null elements (0 null)
    0 / x0 (2): 'a' to 'b'
    1 / x1 (3): 0.5 to 2.0
    array([[1.5, 2.5, 3. ],
           [4.5, 5.5, 6. ]])

    Out-of-bound handling (nan by default, but can be changed via left, right)

    >>> b.interp_axis([-33, 1.5, 44], axis=1, left=-3.3, right=-4.4)
    dimarray: 6 non-null elements (0 null)
    0 / x0 (2): 'a' to 'b'
    1 / x1 (3): -33.0 to 44.0
    array([[-3.3,  2.5, -4.4],
           [-3.3,  5.5, -4.4]])
    """
    pos, name = self._get_axis_info(axis)
    newaxis = Axis(values, name) # necessary array & type checks 

    # sort the axis if needed, to apply numpy interp
    obj = _interp_internal_maybe_sort(self, axis, issorted)
    curaxis = obj.axes[axis]

    # use numpy's built-in for ndim == 1
    if obj.ndim <= 1:
        newval = _numpy_interp(newaxis.values, curaxis.values, obj.values, left=left, right=right)
        newaxes = Axes([newaxis])

    # otherwise calculate linear weights, and re-use them
    else:
        kwargs = _interp_internal_get_weights(curaxis.values, newaxis.values)
        newval = _interp_internal_from_weight(obj.values, axis=pos, left=left, right=right, **kwargs)
        newaxes = [ax.copy() if ax.name != newaxis.name else newaxis for ax in obj.axes]

    dima = obj._constructor(newval, newaxes)
    dima.attrs.update(obj.attrs) # add metadata

    return dima

def interp_like(self, other, **kwargs):
    """Successive application of interp_axis to match another DimArray or axes shape

    Examples
    --------
    >>> from dimarray import DimArray
    >>> a = DimArray([3,4], axes=[[1,3]], dims=['x1'])
    >>> b = DimArray([[1,2,3],[4,5,6]], axes=[['a','b'], [1,2,3]], dims=['x0','x1'])
    >>> a.interp_like(b)
    dimarray: 3 non-null elements (0 null)
    0 / x1 (3): 1 to 3
    array([3. , 3.5, 4. ])
    """
    if hasattr(other, 'axes'):
        axes = other.axes
    elif isinstance(other, Axes):
        axes = other
    else:
        raise TypeError('expected DimArray or Axes, got {}: {}'.format(type(other), other))

    newdims = [ax2.name for ax2 in axes]
    obj = self
    for ax in self.axes:
        if ax.name in newdims:
            newaxis = axes[ax.name].values
            obj = obj.interp_axis(newaxis, axis=ax.name, **kwargs)
    return obj

# Groupy: this is not really a multi-dimensional operation, so we could just remove it.
# dima.to_pandas().groupby(...) is more instructive anyway because of the display.
# def groupby(self, by):
#     """groupby method similar to pandas (only work for 1-D array - see DimArray.flatten)
#
#     Parameters
#     ----------
#     by : array-like (currently 1-D only)
#     
#     Returns
#     -------
#     GroupBy instance
#     """
#     by = np.asarray(by)
#     if self.shape != by.shape:
#         raise ValueError('shape mismatch')
#     by = by.flatten()
#     # groups = itertools.groupby(self.values.flatten()[sorter], lambda by[sorter].tolist())
#     # groups = [(values[sorter[slice_.start]], sorter[slice_]) for slice_ in slices]
#     return GroupBy(self.values.flatten(), by)
#
# class GroupBy(object):
#     def __init__(self, a, by):
#         self.a = a
#         self.sorter = by.argsort()
#         self.by = by
#     def _iter(self):
#         return itertools.groupby(self.sorter, lambda i : self.by[i])
#     @property
#     def groups(self):
#         return {g:list(vals) for g, vals in self._iter()}
#     def apply(self, func, *args, **kwargs):
#         from dimarray import DimArray
#         axis, vals = zip(*[(lab, func(self.a[list(idx)], *args, **kwargs)) \
#                          for lab, idx in self._iter()])
#         return DimArray(np.array(vals), [np.array(axis)])
