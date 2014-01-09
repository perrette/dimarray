""" Provide additional transformations

Requirements for a transformation:

    - call signature: 
	fun(dim1, dim2, ..., dimn, values, *args, **kwargs)

    - returns an instance of Dimarray 

"""
# used in the spatial transforms
from dimarray import apply_recursive, DimArray
from geoarray import GeoArray

from decorators import dimarray_recursive

import region
import grid
import numpy as np

# import descriptors to add methods to Dimarray
#from descriptors import MethodDesc, NumpyDesc, RecDesc, search_fun

#
#   Numpy transformation
#
#
# Regional transforms
#
@dimarray_recursive
def regional_mean(lon, lat, values, myreg=None):
    """ Make a regional mean and get a time series

    Examples:
    ---------
    >>> lon = [0, 50., 100.]
    >>> lat = [0, 50.]
    >>> obj = GeoArray([[1, 2, 4],[ 4, 5, 6]], lon=lon, lat=lat)
    >>> regional_mean(obj)
    """
    myreg = region.check(myreg)
    val = myreg.mean(lon,lat, values)
    return GeoArray(val)

@dimarray_recursive
def interpolate1x1(lon, lat, values, lon0=0.):
    """ Make a regional mean and get a time series
    """
    newlon, newlat, newvalues = grid.interpolate_1x1(lon, lat, values, lon0=lon0)
    return GeoArray(newvalues, lat=newlat, lon=newlon)

@dimarray_recursive
def sample_line(lon, lat, values):
    """ regional sample along a region
    """
    path = zip(lon, lat)
    data = np.zeros(len(path))
    for k, pt in enumerate(path):
	i, j = region.Point(*pt).locate(x, y) # lon, lat
	data[k] = values[i, j]

    data = np.array(data) # make it an array
    return GeoArray(newvalues, ('x',path))

@dimarray_recursive
def extract_box(lon, lat, values, myreg=None):
    """ extract a subregion
    """
    # interactive drawing of a region
    if myreg is None:
	myreg = region.drawbox()

    # check the arguments: initialize a BoxRegion is not yet a region
    myreg = region.check(myreg)

    i, j = myreg.box_idx(lon, lat)
    data = values[i,j]
    lo, la = lon[j], lat[i]
    return GeoArray(data, [('lat',la),('lon',lo)])

@dimarray_recursive
def rectify_longitude(lon, values, lon0):
    """ fix the longitude 
    """
    lon, values = grid.rectify_longitude_data(lon, values, lon0)
    return GeoArray(values, [('lon',lon)])

#
# time transforms
#
def time_mean(obj, period=None):
    """ temporal mean or just slice
    """
    if type(period) is tuple:
	period = slice(*period)

    if period is not None:
	obj = obj.take(period, axis='time', keepdims=True)

    return obj.mean('time')


def since(obj, refperiod):
    """ express w.r.t. a ref period
    """
    return obj - time_mean(obj, refperiod)

def between(obj, refperiod, endperiod):
    """ Make a projection from a refperiod (2 dates) to a refperiod (2 dates)
    """
    obj = obj.since(refperiod)
    return time_mean(obj, endperiod)


#
# ADD TO GEOARRAY
#
