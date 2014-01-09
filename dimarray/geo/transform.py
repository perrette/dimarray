""" Provide additional transformations

Requirements for a transformation:

    - call signature: 
	fun(dim1, dim2, ..., dimn, values, *args, **kwargs)

    - returns an instance of Dimarray 

"""
import numpy as np

# used in the spatial transforms
from dimarray import apply_recursive, DimArray
from geoarray import GeoArray

from decorators import dimarray_recursive

import region as regmod
import grid

# import descriptors to add methods to Dimarray
#from descriptors import MethodDesc, NumpyDesc, RecDesc, search_fun

#
#   Numpy transformation
#
#
# Regional transforms
#
def regional_mean(a, region=None):
    """ Average along lon/lat

    input:
	- a: DimArray instance including lat, lon axis
	- region, optional: 
	    [lon, lat]: Point
	    [lon_ll, lat_ll, lon_ur, lat_ur]: BoxRegion
	    or Region instance

    output:
	- GeoArray instance, regional average

    Notes:
    ------
    Will adjust longitude axis over [0, 360] or [-180, 180] as necessary

    Examples:
    ---------
    >>> lon = [0, 50., 100.]
    >>> lat = [0, 50.]
    >>> a = GeoArray([[1, 2, 4],[ 4, 5, 6]], lon=lon, lat=lat)
    >>> regional_mean(a)
    """
    regobj = regmod.check(region)

    dims = ('lat','lon')

    a = GeoArray(a) # make it a GeoArray to check lon, lat 

    if not set(dims).issubset(a.dims):
	raise ValueError("does not have lon, lat axes: {}".format(a))

    # rearrange dimensions with dims first
    flatdims = [ax.name for ax in a.axes if ax.name not in dims]
    newdims = dims + tuple(flatdims)
    if a.dims != newdims:
	a = a.transpose(newdims)

    # If 2-D return a scalar
    if a.dims == ('lat', 'lon'):
	return regobj.mean(a.lon,a.lat, a.values)

    # flatten all except lat, lon, and make it the first axis
    loopdim = 2
    if len(flatdims) > 1:
	a = a.group(flatdims, insert=loopdim)
	grouped = True
    else:
	grouped = False
    grpaxis = a.axes[loopdim] # grouped axis
    
    # iterate over the first, grouped axis
    results = []
    for k, val in a.iter(loopdim):
	o = val.transpose(('lat','lon'))
	res = regobj.mean(o.lon,o.lat, o.values)
	results.append(res)
	
    # flat object
    average = a._constructor(results, [grpaxis])

    if grouped:
	average = average.ungroup() # now ungroup

    return average

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
	i, j = regmod.Point(*pt).locate(x, y) # lon, lat
	data[k] = values[i, j]

    data = np.array(data) # make it an array
    return GeoArray(newvalues, ('x',path))

@dimarray_recursive
def extract_box(lon, lat, values, region=None):
    """ extract a subregion
    """
    # interactive drawing of a region
    if region is None:
	region = regmod.drawbox()

    # check the arguments: initialize a BoxRegion is not yet a region
    regobj = regmod.check(region)

    i, j = regobj.box_idx(lon, lat)
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

    >>> a = GeoArray([1,2,3],time=[1950, 1955, 1960])
    >>> time_mean(a)
    2.0
    >>> time_mean(a, period=(1955,1960))
    2.5
    """
    if type(period) is tuple:
	period = slice(*period)

    if period is not None:
	obj = obj.take(period, axis='time', keepdims=True)

    return obj.mean('time')


def since(obj, refperiod):
    """ express w.r.t. a ref period

    Examples:
    ---------
    >>> a = GeoArray([1,2,3],time=[1950, 1955, 1960])

    >>> since(a, 1955)
    geoarray: 3 non-null elements (0 null)
    dimensions: 'time'
    0 / time (3): 1950 to 1960
    array([-1.,  0.,  1.])

    >>> since(a, refperiod=(1950,1955)) # a tuple is interpreted as a slice
    geoarray: 3 non-null elements (0 null)
    dimensions: 'time'
    0 / time (3): 1950 to 1960
    array([-0.5,  0.5,  1.5])

    >>> a = GeoArray([[1,2],[3,4],[5,6]],time=[1950, 1955, 1960], lon=[30., 45.]) # multidimensional
    >>> since(a, 1955)
    geoarray: 6 non-null elements (0 null)
    dimensions: 'time', 'lon'
    0 / time (3): 1950 to 1960
    1 / lon (2): 30.0 to 45.0
    array([[-2., -2.],
           [ 0.,  0.],
           [ 2.,  2.]])
    """
    return obj - time_mean(obj, refperiod)

def between(obj, refperiod, endperiod):
    """ Make a projection from a refperiod (2 dates) to a refperiod (2 dates)


    >>> a = GeoArray([1,2,3],time=[1950, 1955, 1960])
    >>> between(a, 1950, 1960)
    2.0
    """
    obj = since(obj, refperiod)
    return time_mean(obj, endperiod)
