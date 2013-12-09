""" Internal API, mostly used to avoid circular imports by lazy import of dependencies, and to keep an overview on what is going on
"""
def dimarray(data, *args, **kwargs):
    """ initialize an Dimarray, avoiding circular import
    """
    from core import dimarray
    #return Dimarray(data, *args, **kwargs)
    return dimarray(data, *args, **kwargs)

def isdimarray(obj):
    """ True is instance of Dimarray or its sub-classes
    """
    from core import Dimarray
    return isinstance(obj, Dimarray)

def axis(*args, **kwargs):
    """ avoid circular imports
    """
    from axes import Axis 
    return Axis(*args, **kwargs)

def axes(*args, **kwargs):
    """ avoid circular imports
    """
    from axes import Axes 
    return Axes(*args, **kwargs)

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
