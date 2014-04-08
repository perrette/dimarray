""" Test module for geo
"""
from dimarray.testing import testmod

import geoarray
import region
import transform
import grid
import decorators

def main(**kwargs):
    """
    """
    testmod(geoarray, **kwargs)
    testmod(region, **kwargs)
    testmod(transform, **kwargs)
    testmod(grid, **kwargs)
    testmod(decorators, **kwargs)

if __name__ == "__main__":
    main()
