""" Test setup
"""
import doctest
import os.path, pkgutil
from importlib import import_module

import core
#from dimarray.core import metadata, core, axes, _indexing as indexing, _transform as transform, _reshape as reshape

class MyTestResults(doctest.TestResults):
    """ test results that can be summed
    """
    maxfailed = 5   # maximum number of failed test before which it stops

    def __add__(self, other):
	assert isinstance(other, doctest.TestResults), "can only sum TestResults instances"
	test = self.__class__(self.failed + other.failed, self.attempted + other.attempted)
        #print 'incremented (self:{}, other:{}, res:{})'.format(self.attempted, other.attempted, test.attempted) # debug
        self.check()
        return test

    def __radd__(self, other):
	return self + other

    def check(self):
	if self.failed > self.maxfailed:
	    raise Exception("Number of failed test {} exceeds max tolerated limit {}".format(self.failed, self.maxfailed))


def get_globals(m=None):
    """ 
    """
    import dimarray
    from dimarray import DimArray
    import dimarray as da
    import numpy as np

    #if hasattr(m, '__all__'):
    #    all_ = m.__all__
    #else:
    if m is not None:
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

    print "...DOCTEST",m.__name__
    return doctest.testmod(m, **kwargs)

def testfile(fname, globs=None, **kwargs):
    if globs is None:
        globs = get_globals() 

    kwargs['globs'] = globs

    print "...DOCTEST",fname
    return doctest.testfile(fname, module_relative=False, **kwargs)

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
#        print dirname
#        root = import_module(dirname.replace(os.path.sep, "."))
#        for f in filenames:
#            if not f.endswith('.py'): 
#                continue
#            if f == "testing.py": continue
#            name, ext = os.path.splitext(f)
#            m = import_module("."+name, package=root.__name__)
#
#            if hasattr(m, 'test') and use_test:
#                m.test(globs=globs, **kwargs)
#            else:
#                testmod(m, globs=globs, **kwargs)


#def test_rec(**kwargs):
#    import dimarray 
#    return testpkg(dimarray, **kwargs)
