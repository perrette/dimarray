import dimarray
import dimarray.core.tests
import dimarray.geo.tests
import dimarray.io.tests

from dimarray.testing import testmod

from dimarray import core
from dimarray import geo
from dimarray import io

def main(**kwargs):
    """
    """
    core.tests.main(**kwargs)
    io.tests.main(**kwargs)
    geo.tests.main(**kwargs)
    testmod(dimarray, **kwargs)

if __name__ == "__main__":
    main()