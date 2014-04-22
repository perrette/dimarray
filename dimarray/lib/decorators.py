import inspect
import functools

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

def _update_doc(cls, *args, **kwargs):
    """ update docstring of an object instance
    """
    clsdict = {k:cls.__dict__[k] for k in cls.__dict__}
    clsdict['__doc__'] = cls.__doc__.format(*args, **kwargs)
    return type(cls.__name__, (cls,), clsdict)

def axes_as_keywords(func):
    """ make a function func(self, values, axis, opt1=.., opt2=..) also accept 
    axes as keyword arguments: func2(self, values, axis, **kwargs)
    where **kwargs is opt1=..., opt2=..., **axes 
    and where **axes is ax1=val1, ax2=val2 etc...
    """
    # arguments of func, including opt1, opt2 etc...
    func_args = inspect.getargspec(func)[0]

    def newfunc(self, values=None, axis=None, **kwargs):
        """ decorated function
        """
        # Separate optional arguments from axes
        kwaxes = {}
        for k in kwargs.keys():
            if k not in func_args:
                kwaxes[k] = kwargs.pop(k)

        # Recursive call if keyword arguments are provided
        if len(kwaxes) > 0:
            assert values is None, "can't input both values/axis and keywords arguments"
            dims = kwaxes.keys()

            # First check that dimensions are there
            for k in dims:
                if k not in self.dims:
                    raise ValueError("can only repeat existing axis, need to reshape first (or use broadcast)")

            # Choose the appropriate order for speed
            dims = [k for k in self.dims if k in dims]
            obj = self
            for k in reversed(dims):
                obj = func(self, kwaxes[k], axis=k, **kwargs)

        else:
            obj = func(self, values, axis=axis, **kwargs)

        return obj

    # update documentation (assume it already includes info about keyword arguments)
    newfunc = functools.update_wrapper(newfunc, func)

    return newfunc
