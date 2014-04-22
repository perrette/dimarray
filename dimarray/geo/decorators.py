import inspect
from dimarray.lib import apply_recursive

def _get_dims(fun):
    """ return the dimensions of a functions based on its call signature

    input:
        function with arguments: dim1, dim2..., dimn, values, *args, **kwargs

    returns:
        [dim1, dim2, ..., dimn]
    """
    args = inspect.getargspec(fun)[0] # e.g. lon, lat, values
    i = args.index('values')
    return args[:i] # all cut-off values (e.g. lon, lat) 

def dimarray_transform(fun):
    """ Decorator

    original function signature:
        fun(dim1, dim2,..., dimn, values, *args, **kwargs)

    new function signature:
        fun(obj, *args, **kwargs)
    """
    # get dimensions from function's signature
    dims = _get_dims(fun)

    from geoarray import GeoArray

    def dimarray_fun(obj, *args, **kwargs):
        """ Automatically generated DimArray function:

        Signature:
            fun(obj, *args, **kwargs)

        Original function signature:
            fun(dim1, dim2,..., dimn, values, *args, **kwargs)

        Original Doc:
        -------------
        """
        obj = GeoArray(obj) # just make sure the order of dimensions if OK, otherwise transpose
        args0 = [obj.axes[k].values for k in dims] # coordinates: e.g. lon, lat...
        args0.append(obj.values)    # add values
        args1 = args0 + list(args)  # add other default arguments

        return fun(*args1, **kwargs)

    dimarray_fun.__name__ += fun.__name__
    dimarray_fun.__doc__ += fun.__doc__

    return dimarray_fun

def dimarray_recursive(fun):
    """ Make a basic function (recursively) applicable on a DimArray

    original:
        fun(dim1, dim2,..., dimn, values, *args, **kwargs)

    new:
        fun(obj, *args, **kwargs)
    """

    # First make it a DimArray ==> DimArray function
    dims = _get_dims(fun)
    fun1 = dimarray_transform(fun)

    def fun_rec(obj, *args, **kwargs):
        return apply_recursive(obj, dims, fun1, *args, **kwargs)

    fun_rec.__name__ = fun1.__name__
    fun_rec.__doc__ = fun1.__doc__

    return fun_rec
