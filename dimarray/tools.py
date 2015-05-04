""" tools
"""
import inspect
import warnings
import numpy as np

# check if an array has any nan
try:
    from bottlebeck import anynan
except ImportError:
    def anynan(a, axis=None):
        """ fast way of checking wether an array has nans

        Parameters
        ----------
         a: numpy array
        """
        if a.size == 0:
            return False
        return np.isnan(a.min(axis=axis))

def pandas_obj(values, *axes):
    """ return a pandas object adapted to the dimensions of values
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("install pandas to use this method")

    pandas_classes = [pd.Series, pd.DataFrame, pd.Panel, pd.Panel4D]

    n = pd.np.array(values).ndim
    if n > len(pandas_classes):
        raise Exception("No pandas class matches that dimension: "+str(n))

    elif n == 0:
        return values

    else:
        cls = pandas_classes[n-1] 
        return cls(values, *axes)

def ndindex(indices, pos):
    """ return the N-D index from an along-axis index

    Parameters
    ----------
        indices: `int` or `list` or `slice`
        pos : axis position
    Returns
    -------
        tuple
    """
    return (slice(None),)*pos + np.index_exp[indices]

def is_DimArray(obj):
    """ avoid import conflict
    """
    from dimarray import DimArray
    return isinstance(obj, DimArray)

def is_array1d_equiv(a):
    " test if a is convertible to a 1-D array of scalar"
    from dimarray import Axis
    if isinstance(a, np.ndarray) and a.ndim == 1:
        res = True
    elif isinstance(a, Axis): 
        res = True
    # [ 1-d array ] is not considered array-equivalent (speed-up axes init)
    elif type(a) is list and len(a) == 1 and is_array1d_equiv(a[0]):
        res = False # np.asarray(a) would return True but it does not matter here
    else:
        try:
            a = np.asarray(a)
            res = a.ndim == 1 and ((a[0] is str) or np.isscalar(a[0]))
        except:
            res = False
    return res

def is_numeric(a):
    " for a ndarray "
    return a.dtype.kind in ("i, f")

# =======================
# A few useful decorators
# =======================
def format_doc(*args, **kwargs):
    """ Apply `format` to docstring
    """
    def pseudo_decorator(func):
        """ not a real decorator as it modifies the function in place
        """
        try:
            func.__doc__ = func.__doc__.format(*args, **kwargs)
        except AttributeError: # non writable
            func.__func__.__doc__ = func.__func__.__doc__.format(*args, **kwargs)
            #print "failed for", func
            #func = _update_doc(func, *args, **kwargs)
        return func

    return pseudo_decorator

def deprecated_func(fun, old_name, msg=None, doc=None):
    """Decorator to deprecate a function
    """
    if msg is None:
        msg = "Function {} is deprecated, use {} instead".format(old_name, fun.__name__)
    def fun_depr(*args, **kwargs):
        warnings.warn(msg, FutureWarning)
        return fun(*args, **kwargs)

    if doc is None:
        doc = "Deprecated. Now renamed to "+fun.__name__ 
    fun_depr.__doc__ = doc
    return fun_depr

# ================
# User-information
# ================
github="github.com/perrette/dimarray"
github_issues="github.com/perrette/dimarray/issues"

def file_an_issue_message():
    return "Please report any problem or feature request at {}".format(github_issues)
