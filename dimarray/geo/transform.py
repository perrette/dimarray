""" Provide additional transformations

Requirements for a transformation:

    - call signature: 
	fun(dim1, dim2, ..., dimn, values, *args, **kwargs)

    - returns an instance of Dimarray 

"""
# used in the spatial transforms
from geotools import region
from geotools import grid
import numpy as np

# import descriptors to add methods to Dimarray
from descriptors import MethodDesc, NumpyDesc, RecDesc, search_fun

from lazyapi import predef, dimarray

#
#   Numpy transformation
#

def numpy_transforms(cls, nan=True):
    """ Class decorator: add numpy transformations as a descriptor

    Transformations below are included (with automatic search):

    numpy_transforms = [ "all", "alltrue", "amax", "amin", "any", "compress",
     "cumprod", "cumproduct", "cumsum", "max", "mean", "median", "min",
     "percentile", "prod", "product", "ptp", "sometrue", "std", "sum", "take", "var"]
    """
    exclude = ["take","amin","amax"]
    numpy_transforms = [f.__name__ for f in search_fun(has=['a','axis','out'], glob=vars(np)) if f.__name__ not in exclude]
    for nm in numpy_transforms:

	# replace with nan-form if applicable
	nm2 = nm
	if nan and hasattr(np, "nan"+nm):
	    nm2 = "nan"+nm

	# Do not overwrite an existing method !!
	if hasattr(cls, nm):
	    print "method already defined, skip:", nm
	    continue

	setattr(cls, nm, NumpyDesc(nm2))
    return cls

#def pandas_transforms(cls):
#    """ add pandas transformation (optional)
#    """
#    import pandas as pd
#    numpy_transforms = search_fun(has=['a','axis','out'], glob=vars(pd))
#    for 

def regional_transforms(cls):
    """ Class decorator: add regional transformations
    """ 
    # get regional transformations
    transforms = search_fun(has=['values'], glob=globals())

    # Now add them to the main Dimarray class
    for f in transforms:
	#print "Add",f.__name__
	#cls.__dict__[f.__name__] = RecDesc(f)
	setattr(cls, f.__name__, RecDesc(f))

    return cls

def extra_transforms(cls):
    """ Class decorator: add a few additional transform method to the class (wrt, between, etc...)
    """
    # other transformation whose first argument is object
    transforms = search_fun(has=['obj'], glob=globals())
    #transforms = ["wrt","between","time_mean"]

    # Now add them to the main Dimarray class
    for f in transforms:
	#print "Add",f.__name__
	#cls.__dict__[f.__name__] = MethodDesc(f)
	setattr(cls, f.__name__, MethodDesc(f))

    return cls

#
# Regional transforms
#

def regional_mean(lon, lat, values, myreg=None):
    """ Make a regional mean and get a time series
    """
    myreg = region.check(myreg)
    val = myreg.mean(lon, lat, values)
    return predef('Scalar',val)

def interpolate1x1(lon, lat, values, lon0=0.):
    """ Make a regional mean and get a time series
    """
    newlon, newlat, newvalues = grid.interpolate_1x1(lon, lat, values, lon0=lon0)
    return predef('Map',newvalues, newlat, newlon)

def sample_line(lon, lat, values):
    """ regional sample along a region
    """
    path = zip(lon, lat)
    data = np.zeros(len(path))
    for k, pt in enumerate(path):
	i, j = region.Point(*pt).locate(x, y) # lon, lat
	data[k] = values[i, j]

    data = np.array(data) # make it an array
    return predef('Line',data, path)

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
    return predef('Map',data, la, lo)

def rectify_longitude(lon, values, lon0):
    """ fix the longitude 
    """
    lon, values = grid.rectify_longitude_data(lon, values, lon0)
    return dimarray(values, [Axis(lon, 'lon')])

#
# time transforms
#
def time_mean(obj, period=None):
    """ temporal mean or just slice
    """
    if period is None:
	return obj.mean('time')

    elif type(period) is not list:
	return obj.xs(time=period)

    else:
	return obj.xs(time=period).mean('time')

def wrt(obj, refperiod):
    """ express w.r.t. a ref period
    """
    return obj - obj.time_mean(refperiod)

def between(obj, refperiod, endperiod):
    """ Make a projection from a refperiod (2 dates) to a refperiod (2 dates)
    """
    return obj.wrt(refperiod).time_mean(endperiod)
