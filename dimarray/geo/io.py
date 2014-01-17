""" Return GeoArray when reading
"""
from dimarray import read_nc as read_nc_da
from geoarray import GeoArray

def read_nc(*args, **kwargs):
    """ like dimarray.read_nc, but return a GeoArray instead
    """
    a = read_nc_da(*args, **kwargs)
    return GeoArray(a)
