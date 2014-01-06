""" Test setup
"""
import doctest
import os.path, pkgutil
from importlib import import_module

import core
#from dimarray.core import metadata, core, axes, _indexing as indexing, _transform as transform, _reshape as reshape

def get_globals(m):
    """ 
    """
    import dimarray
    from dimarray import DimArray
    import dimarray as da
    import numpy as np
    #if hasattr(m, '__all__'):
    #    all_ = m.__all__
    #else:
    all_ = [k for k in dir(m) if not k.startswith("_")]
    for k in all_:
	locals()[k] = getattr(m,k)
    return locals()

def testmod(m, globs=None, **kwargs):
    """ test all docstrings of a module
    """
    if globs is None:
	globs = get_globals(m) 

    kwargs['globs'] = globs

    print "\n\n============================"
    print "TEST",m.__name__
    print "============================\n\n"
    doctest.testmod(m, **kwargs)

#def testpkg(pkg, globs=None, use_test=True, **kwargs):
#    """ test all docstrings in a package by recursively testing submodules
#
#    use_test: if True (default), use local test() function instead of directly testmod
#    """
#    pkgpath = os.path.dirname(pkg.__file__)
#
#    # list of directory and filenames
#    for dirname,dirnames,filenames in os.walk(pkgpath):
#
#	print dirname
#	root = import_module(dirname.replace(os.path.sep, "."))
#	for f in filenames:
#	    if not f.endswith('.py'): 
#		continue
#	    if f == "testing.py": continue
#	    name, ext = os.path.splitext(f)
#	    m = import_module("."+name, package=root.__name__)
#
#	    if hasattr(m, 'test') and use_test:
#		m.test(globs=globs, **kwargs)
#	    else:
#		testmod(m, globs=globs, **kwargs)


#def test_rec(**kwargs):
#    import dimarray 
#    return testpkg(dimarray, **kwargs)

def test_all(**kwargs):
    """
    """
    import core.metadata as metadata
    import core.core as core
    import core.axes as axes
    import core._indexing as indexing
    import core._reshape as reshape
    import core._transform as transform

    testmod(metadata, **kwargs)

    testmod(core, **kwargs)

    testmod(axes, **kwargs)

    testmod(indexing, **kwargs)

    testmod(transform, **kwargs)

    testmod(reshape, **kwargs)
