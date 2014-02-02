import dimarray
import dimarray.core.tests
import dimarray.geo.tests
import dimarray.io.tests

from dimarray.testing import testmod, testfile

from dimarray import core
from dimarray import geo
from dimarray import io
from dimarray import dataset

def main(**kwargs):
    """
    """
    core.tests.main(**kwargs)
    io.tests.main(**kwargs)
    geo.tests.main(**kwargs)
    testmod(dimarray, **kwargs)
    testmod(dataset, **kwargs)
    try:
	testfile('README.rst')
    except:
	print 'README.rst not found'

if __name__ == "__main__":
    main()
