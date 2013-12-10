""" Internal API, mostly used to avoid circular imports by lazy import of dependencies, and to keep an overview on what is going on
"""
def array(data, *args, **kwargs):
    """ initialize an DimArray, avoiding circular import
    """
    from core import array
    #return DimArray(data, *args, **kwargs)
    return array(data, *args, **kwargs)

def isdimarray(obj):
    """ True is instance of DimArray or its sub-classes
    """
    from core import DimArray
    return isinstance(obj, DimArray)

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
