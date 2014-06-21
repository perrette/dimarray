""" a few useful decorators
"""
def format_doc(*args, **kwargs):
    """ Apply `format` to docstring
    """
    def pseudo_decorator(func):
        """ not a real decorator as it modifies the function in place
        """
        try:
            func.__doc__ = func.__doc__.format(*args, **kwargs)
        except AttributeError: # non writable
            print "failed for", func
            func = _update_doc(func, *args, **kwargs)
        return func

    return pseudo_decorator


def many(func):
    """ decorator to make a function applicable to a list
    """
    def newfunc(v, *args, **kwargs):
        if np.isiterable(v):
            return [func(vi, *args, **kwargs) for vi in v]
        else:
            return func(vi, *args, **kwargs)
    newfunc.__doc__ = func.__doc__
    return newfunc


def apply_multiindex(obj, function, ix, args=(), **kwargs):
    """ decorator to make a function work with multi-indices, by chained call

    Parameters
    ----------
    obj: DimArray object
    function: function to be chained-called
    ix: indices, tuple or dict
    *args: variable arguments to function
    **kwargs: keyword arguments to function

    Returns
    -------
    DimArray object or scalar
    """
    # if indices is a numpy multiindex 
    if type(ix) is tuple:
        ix = {obj.axes[i].name: val for i, val in enumerate(ix)}

    # now we have a dict
    assert isinstance(ix, dict), "multi-indexing in transforms only works with tuple or dict"

    for k in ix:
        kwargs["axis"] = k
        obj = function(obj, ix[k], **kwargs)

    return obj

def multiindex(func):
    def newfunc(obj, values, axis=0, **kwargs):
        # deal with multi-indexing
        if type(values) in tuple or isinstance(values, dict):
            assert axis is None or axis == 0, "cannot have axis > 0 with tuple values"
            return apply_multiindex(obj, func, values, **kwargs)
        else:
            return func(obj, values, axis=axis, **kwargs)
    
    newfunc.__doc__ = func.__doc__
    return newfunc
