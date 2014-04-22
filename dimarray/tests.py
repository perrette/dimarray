from warnings import warn

import dimarray
import dimarray.core.tests
import dimarray.io.tests
import dimarray.lib.tests

from dimarray.testing import testmod, testfile

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
    core.tests.main(**kwargs)
    io.tests.main(**kwargs)
    if geo_success: geo.tests.main(**kwargs)
    testmod(dimarray, **kwargs)
    testmod(dataset, **kwargs)
    #testmod(lib, **kwargs)
    lib.tests.main(**kwargs)
    dataset.test()
    try:
        testfile('README.rst')
    except:
        print 'README.rst not found'

if __name__ == "__main__":
    main()
