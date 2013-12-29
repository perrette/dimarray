""" various tools
"""
import inspect

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

    inputs:
	indices: `int` or `list` or `slice`
	pos : axis position
    returns:
	tuple
    """
    return (slice(None),)*pos + np.index_exp[indices]

def is_Dimarray(obj):
    """ avoid import conflict
    """
    from core import Dimarray
    return isinstance(obj, Dimarray)

