import functools

def format_doc(*args, **kwargs):
    """ Apply `format` to docstring
    """
    def pseudo_decorator(func):
	""" not a real decorator as it modifies the function in place
	"""
	func.__doc__ = func.__doc__.format(*args, **kwargs)
	return func

    return pseudo_decorator

class Desc(object):
    """ Convert a function on Dimarray into a method
    """
    def __init__(self, func):
	""" func: function taking "obj" as first argument
	"""
	self.func = func

    def __get__(self, obj, cls=None):
	""" return a bound method with updated documention
	"""
	method = functools.partial(obj=self.func, **self.kwargs)
	method = functools.update_wrapper(method, self.func)

	# Remove reference to obj
	patt = 'obj: Dimarray instance\n'
	if patt+'\n' in method.__doc__:
	    method.__doc__ = method.__doc__.replace(patt+'\n','')
	else:
	    method.__doc__ = method.__doc__.replace(patt,'')

	return method
