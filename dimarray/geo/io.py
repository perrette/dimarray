""" Return GeoArray when reading
"""
import dimarray as da # for the docstr
from dimarray import read_nc as read_nc_da
from geoarray import GeoArray

__all__ = ['read_nc']

def read_nc(*args, **kwargs):
    """ like dimarray.read_nc, but return a GeoArray instead
    """
    a = read_nc_da(*args, **kwargs)
    return GeoArray(a)

# update the doc
read_nc.__doc__ = read_nc_da.__doc__.replace('DimArray','GeoArray')
