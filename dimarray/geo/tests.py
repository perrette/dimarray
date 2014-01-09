""" Test module for geo
"""
from dimarray.testing import testmod

import geoarray
import region
import transform
import grid
import decorator

def main(**kwargs):
    """
    """
    testmod(geoarray, **kwargs)
    testmod(region, **kwargs)
    testmod(transform, **kwargs)
    testmod(grid, **kwargs)
    testmod(decorator, **kwargs)

if __name__ == "__main__":
    main()
