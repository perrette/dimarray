from warnings import warn

import dimarray
import dimarray.core.tests
import dimarray.io.tests
import dimarray.lib.tests

from dimarray.testing import testmod, testfile, MyTestResults

from dimarray import core
from dimarray import io
from dimarray import dataset
from dimarray import lib

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
    test += core.tests.main(**kwargs)
    io.tests.main(**kwargs)
    if geo_success: test += geo.tests.main(**kwargs)
    test += testmod(dimarray, **kwargs) 
    test += testmod(dataset, **kwargs)   
    #testmod(lib, **kwargs)
    lib.tests.main(**kwargs)
    try:
        test += testfile('README.rst')
    except IOError:
        print 'README.rst not found'

    return test

if __name__ == "__main__":
    main()
