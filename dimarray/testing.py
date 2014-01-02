""" Test setup
"""

import doctest

def get_globals():
    """ 
    """
    import dimarray as da
    return locals()

def testmod(m, globs=None, **kwargs):
    """ test all docstrings of a module
    """
    if globs is None:
	globs = get_globals() 

    kwargs['globs'] = globs
    doctest.testmod(m, **kwargs)
