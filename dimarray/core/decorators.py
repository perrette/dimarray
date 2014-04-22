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
