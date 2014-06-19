""" Test setup
"""
import doctest
import os.path, pkgutil
import inspect
from types import ModuleType
from importlib import import_module
from collections import OrderedDict as odict

import dimarray

#from dimarray.core import metadata, core, axes, _indexing as indexing, _transform as transform, _reshape as reshape

class MyTestResults(doctest.TestResults):
    """ test results that can be summed
    """
    maxfailed = 5   # maximum number of failed test before which it stops

    def __add__(self, other):
	assert isinstance(other, doctest.TestResults), "can only sum TestResults instances"
	test = self.__class__(self.failed + other.failed, self.attempted + other.attempted)
        #print 'attempted add (self:{}, other:{}, res:{})'.format(self.attempted, other.attempted, test.attempted) # debug
        self.check()
        return test

    def __radd__(self, other):
	return self + other

    def check(self):
	if self.failed > self.maxfailed:
	    raise Exception("Number of failed test {} exceeds max tolerated limit {}".format(self.failed, self.maxfailed))

    def summary(self):
        """ summary of tests
        """
        print "============================="
        print "Failed   : {}\nAttempted: {}".format(self.failed, self.attempted)
        print "============================="
        if self.failed > 0:
            print ">>>>>>>>> Failed Tests"


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

def testmod(m, **kwargs):
    """ test all docstrings of a module
    """
    if 'globs' not in kwargs:
        kwargs['globs'] = get_globals(m)

    print "...DOCTEST:",m.__name__
    return doctest.testmod(m, **kwargs)

def testfile(fname, globs=None, **kwargs):
    if globs is None:
        globs = get_globals() 

    kwargs['globs'] = globs

    print "...DOCTEST:",fname
    return doctest.testfile(fname, module_relative=False, **kwargs)


#
# Try to look recursively for modules to test for doctests
#
def _issubmodule(m1, m2):
    " test if m1 is submodule of m2 "
    return m1.__name__[:len(m2.__name__)+1] == m2.__name__+'.'

def _get_submodules(m):
    """ get submodules present in a module's global
    """
    return inspect.getmembers(m, lambda x: isinstance(x, ModuleType) and _issubmodule(x, m))

class MyDocTest(object):
    """ class to help recursively looking for submodules
    """
    tested_modules = odict() # keep individual tested modules
    result = MyTestResults(0, 0) # start at 0

    def __init__(self, module):
        self.module = module

    def testmod(self, **kwargs):

        # make sure the test is not run twice
        if self.module.__name__ in self.tested_modules:
            print 'Warning: ',self.module.__name__+' module already tested'
            return self.tested_modules[self.module.__name__]

        #if 'globs' not in kwargs:
        #    kwargs['globs'] = get_globals(self.module)
        #print "...DOCTEST:",self.module.__name__
        #res = doctest.testmod(self.module, **kwargs)

        res = testmod(self.module, **kwargs)

        # document all doc-tests
        self.tested_modules[self.module.__name__] = res
        self.__class__.result += res  # add test results

        return res

    def recursive_testmod(self):
        self.testmod() # test current module
        for name, mod in _get_submodules(self.module):

            if name in self.tested_modules:
                print 'module already tested', mod.__name__
                continue

            MyDocTest(mod).recursive_testmod()

    @classmethod
    def summary(cls, verbose=False):
        if verbose:
            print "All tested modules:"
            print cls.tested_modules.keys()
            print "...detailed results"
            print cls.tested_modules
        return cls.result.summary()

def main():
    MyDocTest(dimarray).recursive_testmod() # recursive testing starting at dimarray
    #MyDocTest(dimarray).recursive_testmod() # recursive testing starting at dimarray

    print "dimarray's doctest"
    MyDocTest.summary()

    # Also test on 
    #totest = ['doctests', 'olddocs', 'test_core', 'testing', 'test_io', 'test_mpl', 'test_operations']
    totest = ['doctests', 'olddocs', 'test_core', 'testing', 'test_io', 'test_mpl', 'test_operations']
    for nm in totest:
        m = import_module(nm)
        MyDocTest(m).testmod()

    print "Including tests directory"
    MyDocTest.summary()

if __name__ == '__main__':
    main()
