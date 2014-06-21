""" various tools
"""
import inspect
import numpy as np

def pandas_obj(values, *axes):
    """ return a pandas object adapted to the dimensions of values
    """
    import pandas as pd
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
    if isinstance(a, np.ndarray) and a.ndim == 1:
        res = True
    else:
        try:
            a = np.asarray(a)
            res = a.ndim == 1 and ((a[0] is str) or np.isscalar(a[0]))
        except:
            res = False
    return res

