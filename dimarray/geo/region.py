""" Region object to extract data from netcdf
"""

from copy import copy
import numpy as np
from grid import rectify_longitude, rectify_longitude_data, LonLatGrid
from dimarray.compat.basemap import interp

__all__ = ["World","Point","BoxRegion","drawbox","extract"] # from region import *

def extract(lon, lat, data, region):
    """ Extract region from the data
    
    Input:
        - lon, lat, data    : data + coordinate
        - region            : region to extract

    Output:
        - lonr, latr, datar : regional data

    Region can be:
        - Single lon/lat location   : lon_p, lat_p tuple
        - Rectangular region            : lon_ll, lat_ll, lon_ur, lat_ur
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

def drawpoly(ax=None):
    """ interactive drawing of a polygone
    """
    import matplotlib.pyplot as plt
    if ax is None: plt.gca()

    print 'Select path'
    xy = plt.ginput(n=0, timeout=0)

    lon = [p[0] for p in xy]
    lat = [p[1] for p in xy]

    return Polygon(lon, lat)


class Region(object):
    """ region class

    public methods: 
        mean : extract data from lon, lat, map
        box_idx: return lat/lon indices of the smallest containing box (useful to read from netCDF)
        plot        : draw the region on a map

    private methods (methods that are not supposed to be used by the user, but used by the public methods)
        contour : return the region contour as a polygone (lons, lats)
        extract : return all data from the region, as an array (1-D or 2-D)
    """
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

        if hasattr(self, 'lon0'):
            lon = rectify_longitude(lon, lon0 = self.lon0)
        lonr, latr, datar = self.extract(lon, lat, data) # extract regional data
        LO, LA = np.meshgrid(lonr, latr)
        w = np.cos(np.radians(LA)) # weights

        if isinstance(datar, np.ma.MaskedArray):
            datar = datar.filled(np.nan)

        w[np.isnan(datar)] = np.nan

        return np.nansum(datar*w)/np.nansum(w)  # weighted mean

    def extract(self, lon, lat, data):
        """ extract data from the region
        """
        if hasattr(self, 'lon0'):
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

class Polygon(Region):
    """
    """
    def __init__(self, lon, lat, lon0=None):
        self.coords = lon, lat
        if lon0 is None:
            lon0 = get_lon0(*lon)
        self.lon0 = lon0

#    def plot(self, **kwargs):
#        import matplotlib.pyplot as plt
#        lon, lat = self.to_shapely().exterior.xy
#        ax = kwargs.pop('ax', plt.gca())
#        return ax.plot(lon, lat, **kwargs)

    def contour(self):
        lons, lats = self.to_shapely().exterior.xy
        return np.asarray(lons), np.asarray(lats)

    def to_shapely(self):
        from shapely.geometry import Polygon
        lon, lat = self.coords
        poly = Polygon(zip(lon, lat))
        #poly = Polygon(np.vstack((lon, lat)).T)
        return poly

    def contains_arrays(self, lon, lat):
        """ return an array of same size as lon and lat

        lon, lat: arrays of longitude and latitude
        """
        from matplotlib.nxutils import points_inside_poly as inpoly
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        lon = rectify_longitude(lon, lon0=self.lon0, sort=False)
        #if lon.ndim == 1:
        #    lon, lat = np.meshgrid(lon, lat)
        xypoints = np.vstack((lon.flatten(), lat.flatten())).T
        xyvertices = np.asarray(self.to_shapely().exterior.xy).T # contours of the polynom
        mask = inpoly(xypoints, xyvertices)
        return mask.reshape(lon.shape)
        #region = np.zeros_like(lon, dtype=bool)

    def contains(self, lon, lat):
        """ return True if x, y is in the Polygon
        """
        if np.size(lon) > 1:
            #return np.array([self.contains(xx, yy) for xx, yy in zip(lon, lat)])
            return self.contains_arrays(lon, lat)

        from shapely.geometry import Point
        poly = self.to_shapely()
        lon = rectify_longitude(lon, lon0=self.lon0, sort=False)
        res = poly.contains(Point(lon, lat))
        assert res is not None
        return res

class Mask(Region):
    """ Region defined by a boolean mask
    """
    def __init__(self, lon, lat, mask, lon0=None):
        """
        """
        assert lon.ndim == 1 and lat.ndim == 1, "can only have 1-D coordinates"
        self.coords = lon, lat
        if lon0 is None:
            lon0 = get_lon0(*lon)
        self.lon0 = lon0
        self.mask = mask

    def interpolate(self, lon, lat):
        """ interpolate the mask

        lon, lat: 1-D arrays to be passed to meshgrid
        """
        assert lon.ndim == 1 and lat.ndim == 1, "must be 1-D coordinates compatible with meshgrid"
        # first fix lon0 to match new data
        lon0 = get_lon0(*lon.flatten())
        mlon, mlat = self.coords
        mlon, mask = rectify_longitude_data(mlon, self.mask, lon0)

        # make input coords is 2-D
        #if lon.ndim == 1 and meshgrid:
        lon2, lat2 = np.meshgrid(lon, lat)

        maskf = interp(np.asarray(self.mask, dtype=float), mlon, mlat, lon2, lat2)
        mask = maskf >= 0.5
        return Mask(lon, lat, mask, lon0)

    def contains(self, lon, lat):
        """ Return a boolean array of same shape as lon and lat with points contained in the patch
        """
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        mlon, mlat = self.coords
        assert lon.shape == lat.shape, "lon, lat must have the same shape"
        #lon = rectify_longitude(lon, lon0=self.lon0, sort=False)

        # first fix lon0 to match new data
        lon0 = get_lon0(*lon.flatten())
        if lon0 != self.lon0:
            self = copy(self)
            mlon, self.mask = rectify_longitude_data(mlon, self.mask, lon0)

        if not (np.all(lon == mlon) and np.all(lat == mlat)):
            self = copy(self)

            # interpolate the mask onto the input grid
            maskf = interp(np.asarray(self.mask, dtype=float), mlon, mlat, lon, lat)
            mask = maskf >= 0.5

        else:
            mask = self.mask

        return mask

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        lon, lat = self.coords
        return ax.contourf(lon, lat, np.asarray(self.mask, dtype=float))

#    def mask(self, lon, lat):
#        """ make a mask from a 2-D lon, lat grid
#        """
#        lon = rectify_longitude(lon, lon0=self.lon0)
#        if lon.ndim == 1:
#            lon, lat = np.meshgrid(lon, lat)
#
#        region = np.zeros_like(lon, dtype=bool)
#
#        ni, nj = region.shape
#        for i in range(ni):
#            for j in range(nj):
#                res = self.contains(lon[i,j], lat[i,j])
#                assert res is not None
#                region[i,j] = self.contains(lon[i,j], lat[i,j])
#
#        assert np.any(region)
#        return region


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
        lon = rectify_longitude(lon, lon0=self.lon0)

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
        i, j = grd.locate_point(*self.coords, **kwargs)
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

