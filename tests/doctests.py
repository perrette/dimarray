from warnings import warn

import dimarray
from tests import test_core, test_io, test_lib

from dimarray import core
from dimarray import io
from dimarray import dataset
from dimarray import lib

from testing import testmod, testfile, MyTestResults

# The geo module is experimental and involves additional packages
try:
    import dimarray.geo.tests
    from dimarray import geo
    geo_success = True

except ImportError, msg:
    warn("could not import geo, probably because of missing packages:\n{}".format(msg))
    geo_success = False

def main(**kwargs):
    """
    """
    test = MyTestResults(0, 0)
    test += test_core()

    test += testmod(dimarray, **kwargs) 
    test += testmod(dataset, **kwargs)   
    if geo_success: test += geo.tests.main(**kwargs)
    #testmod(lib, **kwargs)
    try:
        test += testfile('README.rst')
    except IOError:
        print 'README.rst not found'

    return test

if __name__ == "__main__":
    main()
