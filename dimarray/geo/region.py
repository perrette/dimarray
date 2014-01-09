""" Region object to extract data from netcdf
"""

import numpy as np
from grid import rectify_longitude, rectify_longitude_data, LonLatGrid

__all__ = ["World","Point","BoxRegion","drawbox","extract"] # from region import *

def extract(lon, lat, data, region):
    """ Extract region from the data
    
    Input:
	- lon, lat, data    : data + coordinate
	- region	    : region to extract

    Output:
	- lonr, latr, datar : regional data

    Region can be:
	- Single lon/lat location   : lon_p, lat_p tuple
	- Rectangular region	    : lon_ll, lat_ll, lon_ur, lat_ur
	- Region object	

    >>> lonr, latr, datar = extract(lon, lat, data, [31.5, 42.])  # doctest: +SKIP
    """
    if not isinstance(region, Region):

	region = get(*region)

    return region.extract(lon, lat, data)


def get(*region):
    """ get a region given list arguments
    """
    if len(region) == 2:
	region = Point(*region)

    elif len(region) == 4:
	region = BoxRegion(*region)

    else:
	raise ValueError("bad arguments to initialize a region: "+repr(region))

    return region

def check(region=None):
    """ check if region is an instance of Region, otherwise return a Region
    """
    if region is None:
	region = World()  #  just make a weighted mean of the data...

    elif not isinstance(region, Region):
	region = get(*region)

    return region

def drawbox(ax=None):
    """ interactive drawing()
    """
    import matplotlib.pyplot as plt
    if ax is None: plt.gca()

    print 'Select lower left and upper right corners'
    xy = plt.ginput(n=2, timeout=0)

    #x = [p[0] for p in xy]
    #y = [p[1] for p in xy]

    return BoxRegion(*(xy[0]+xy[1]))


class Region(object):
    """ region class

    public methods: 
	mean : extract data from lon, lat, map
	box_idx: return lat/lon indices of the smallest containing box (useful to read from netCDF)
	plot	: draw the region on a map

    private methods (methods that are not supposed to be used by the user, but used by the public methods)
	contour : return the region contour as a polygone (lons, lats)
	extract : return all data from the region, as an array (1-D or 2-D)
    """
    def __init__(self, *coords):
	self.coords = coords
	self.lon0 = get_lon0(lon)

    def plot(self, *args, **kwargs):
	""" plot a region

	same arguments as plot, except...
	lon0: if provided, adjust the data to have the right alignment...
	"""
	import matplotlib.pyplot as plt
	lons, lats = self.contour()
	if 'lon0' in kwargs:
	    lons = rectify_longitude(lons, lon0 = kwargs.pop('lon0'))

	return plt.plot(lons, lats, *args, **kwargs)

    def mean(self, lon, lat, data):
	""" return average data over the region (weighted mean)

	input: 
	    lon, lat, data

	returns: 
	    regional mean of data
	"""
	if np.ndim(lon) == 2:
	    raise Exception('must be a regular grid !')

	lon = rectify_longitude(lon, lon0 = self.lon0)
	lonr, latr, datar = self.extract(lon, lat, data) # extract regional data
	LO, LA = np.meshgrid(lonr, latr)
	w = np.cos(LA) # weights

	if isinstance(datar, np.ndarray):
	    datar = np.ma.array(datar, mask=np.isnan(datar))
	return np.sum(datar*w)/np.sum(w)  # weighted mean

    def extract(self, lon, lat, data):
	""" extract data from the region
	"""
	lon = rectify_longitude(lon, lon0 = self.lon0)
	ii, jj = self.box_idx(lon, lat) # for a rectangular box, box_idx exactly contains the region
	return lon[jj], lat[ii], data[ii, jj]

def get_lon0(*lons):
    """ set lon0
    """
    lon0 = 0.
    for lon in lons:
	if lon < lon0:
	    lon0 = -180.
	    break

    return lon0

class BoxRegion(Region):
    """ Rectangular region
    """
    def __init__(self, lon_llcorner, lat_llcorner, lon_urcorner, lat_urcorner):
	self.coords = lon_llcorner, lat_llcorner, lon_urcorner, lat_urcorner

	self.lon0 = get_lon0(lon_llcorner)

    def contour(self):
	""" return contour of the rectangle (trigonometric direction)
	"""
	lon_llcorner, lat_llcorner, lon_urcorner, lat_urcorner = self.coords
	lons = np.array([lon_llcorner, lon_urcorner, lon_urcorner, lon_llcorner, lon_llcorner])
	lats = np.array([lat_llcorner, lat_llcorner, lat_urcorner, lat_urcorner, lat_llcorner])
	return lons, lats

    def box_idx(self, lon, lat):
	""" return lat/lon index slices of the smallest containing box 
	
	Designed to read from netCDF 

	input:
	    lon, lat: 1-D arrays (must be a regular grid)

	output:
	    ii, jj: slice
	"""
	lon = rectify_longitude(lon, lon0 = self.lon0)

	lon_llcorner, lat_llcorner, lon_urcorner, lat_urcorner = self.coords
	ii = (lat >= lat_llcorner) & (lat <= lat_urcorner)
	jj = (lon >= lon_llcorner) & (lon <= lon_urcorner)

	# find indices
	ii, jj = np.where(ii)[0], np.where(jj)[0]

	# make it a slice
	ii = slice(ii[0], ii[-1]+1)
	jj = slice(jj[0], jj[-1]+1)

	return ii, jj


class World(Region):
    """ World : keep everything
    """
    def box_idx(self, lon, lat):
	return slice(None), slice(None)

class Point(Region):
    """ only a single location
    """
    def __init__(self, lon, lat):
	self.coords = lon, lat
	self.lon0 = get_lon0(lon)

    def contour(self):
	return self.coords

    def locate(self, lon, lat, **kwargs):
	""" Locate a point on a grid
	"""
	grd = LonLatGrid(lon, lat)
	i, j = grd.locate_point(self.lon, self.lat, **kwargs)
	return i, j

    def box_idx(self, lon, lat):
	""" 
	"""
	return self.locate(lon, lat)

    def plot(self, *args, **kwargs):
	import matplotlib.pyplot as plt
	lon, lat = self.coords
	if 'marker' not in kwargs:
	    kwargs['marker'] = 'o'
	plt.plot(lon, lat, *args, **kwargs)


