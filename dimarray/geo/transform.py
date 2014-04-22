""" Provide additional transformations

Requirements for a transformation:

    - call signature: 
        fun(dim1, dim2, ..., dimn, values, *args, **kwargs)

    - returns an instance of Dimarray 

"""
import numpy as np
import warnings

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
    >>> a = GeoArray([[1, 2, 4],[4, 5, 6]], lon=lon, lat=lat)
    >>> rm = regional_mean(a)
    >>> rm
    3.3767428905946684

    Which just weights data with cos(lat)
    >>> lon2, lat2 = np.meshgrid(lon, lat)
    >>> w = np.cos(np.radians(lat2))
    >>> rm2 = np.sum(a.values*w)/np.sum(w)  
    >>> round(rm2, 4)
    3.3767

    GeoArray detects the weights automatically with name "lat"
    >>> rm3 = a.mean() 
    >>> round(rm3,4)
    3.3767
    >>> a.mean(weights=None)
    3.6666666666666665

    Also works for multi-dimensional data shaped in any kind of order
    >>> a = a.newaxis('z',[0,1,2])
    >>> a = a.transpose(('lon','z','lat'))
    >>> regional_mean(a)
    geoarray: 3 non-null elements (0 null)
    dimensions: 'z'
    0 / z (3): 0 to 2
    array([ 3.37674289,  3.37674289,  3.37674289])
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
    agrp = a.group(flatdims, insert=0) 
    grpaxis = agrp.axes[0] # grouped axis
    
    # iterate over the first, grouped axis
    results = []
    for i in range(grpaxis.size):
        res = regobj.mean(a.lon,a.lat, agrp.values[i])
        results.append(res)
        
    # flat object
    grp_ave = a._constructor(results, [grpaxis])

    average = grp_ave.ungroup() # now ungroup

    return average

@dimarray_recursive
def interpolate_1x1(lon, lat, values, lon0=0.):
    """ Make a regional mean and get a time series
    """
    newlon, newlat, newvalues = grid.interpolate_1x1(lon, lat, values, lon0=lon0)
    return GeoArray(newvalues, lat=newlat, lon=newlon)

#def sample_line(a, path):
#    """ regional sample along a region
#
#    path: list of lon/lat points to sample
#
#    >>> lon = np.linspace(0, 50., 100)
#    >>> lat = np.linspace(30, 60., 100)
#    >>> lon2, lat2 = np.meshgrid(lon, lat)
#    >>> a = DimArray(lat2, [('lat',lat),('lon',lon)])
#    >>> sample_line(a, path=[(5, 34), (10, 40), (12,50)])
#    """
#    #path = zip(lon, lat)
#    # locate numpy indices first
#    indices = []
#    for k, pt in enumerate(path):
#        i, j = regmod.Point(*pt).locate(a.lon, a.lat) # lon, lat
#        indices.append((i,j)) # approximate location
#
#    ilat, ilon = zip(*indices)
#
#    return a.take({'lon':ilat, 'lat':ilon}, matlab_like=False, newaxis='x')

def extract_box(a, region=None):
    """ extract a subregion

    a        : DimArray instance
    region: lon_ll, lat_ll, lon_ur, lat_ur 
        or Region instance
    """
    # interactive drawing of a region
    if region is None:
        region = regmod.drawbox()

    # check the arguments: initialize a BoxRegion if not yet a region
    regobj = regmod.check(region)

    i, j = regobj.box_idx(a.lon, a.lat)
    return a.take({'lat':i, 'lon':j}, indexing='position')
    #return GeoArray(data, [('lat',la),('lon',lo)])

@dimarray_recursive
def shift_longitude(lon, values, lon0):
    """ Shift longitude axis, making it start at lon0
    """
    lon, values = grid.rectify_longitude_data(lon, values, lon0)
    return GeoArray(values, [('lon',lon)])

rectify_longitude = shift_longitude # back-compat

#
# time transforms
#
def check_dim(obj, dim, msg=None, raise_error=False):
    if not dim in obj.dims:
        if msg is None: 
            msg = "{} dimension missing, cannot apply this transformation".format(msg)
        if raise_error: 
            raise ValueError(msg)
        else:
            warnings.warn(msg)
        return False
    else:
        return True

def time_mean(obj, period=None, skipna=False):
    """ temporal mean or just slice

    >>> a = GeoArray([1,2,3],time=[1950, 1955, 1960])
    >>> time_mean(a)
    2.0
    >>> time_mean(a, period=(1955,1960))
    2.5
    """
    if not check_dim(obj, 'time'): return obj

    if type(period) is tuple:
        period = slice(*period)

    if period is not None:
        obj = obj.take(period, axis='time', keepdims=True)

    return obj.mean('time', skipna=skipna)


def since(obj, refperiod, skipna=False):
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
    if not check_dim(obj, 'time'): return obj
    return obj - time_mean(obj, refperiod, skipna=skipna)

def between(obj, refperiod, endperiod, skipna=False):
    """ Make a projection from a refperiod (2 dates) to a refperiod (2 dates)


    >>> a = GeoArray([1,2,3],time=[1950, 1955, 1960])
    >>> between(a, 1950, 1960)
    2.0
    """
    if not check_dim(obj, 'time'): return obj
    obj = since(obj, refperiod, skipna=skipna)
    return time_mean(obj, endperiod, skipna=skipna)
