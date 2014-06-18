""" Test module for geo
"""
from dimarray.testing import testmod, MyTestResults

import geoarray
import region
import transform
import grid
import decorators

def main(**kwargs):
    """
    """
    test = MyTestResults(0, 0)
    test += testmod(geoarray, **kwargs)
    test += testmod(region, **kwargs)
    test += testmod(transform, **kwargs)
    test += testmod(grid, **kwargs)
    test += testmod(decorators, **kwargs)
    return test

if __name__ == "__main__":
    main()
