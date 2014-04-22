""" Routine to deal with spatial interpolation, looking up coordinates etc...
"""

import numpy as np

def get_lon0(x):
    """
    """
    if x.min() < 0: lon0=-180.
    else: lon0 = 0.
    return lon0

class LonLatGrid(object):
    """ Longitude / Latitude grid
    """
    def __init__(self, x, y, lon0=None):
        """ initialize the grid 
        x, y: input array (1-D or 2-D), assume that the data it refers to 
        has y as first dimension and x as second (cartesian coordinate convention)
        """
        if np.ndim(x) == 1:
            x, y = np.meshgrid(x, y)

        self.x = x
        self.y = y

        # automatically find an appropriate reference for the grid
        lon0 = get_lon0(x)

        self.lon0 = lon0 # starting point for the longitude
        self.coslat = np.cos(y/180.*np.pi)

    def locate_point(self, lon1, lat1, maxdist=None, mask=None):
        """ find the grid indices corresponding 
        """
        LON, LAT = self.x, self.y

        # do the calculation on [lon0, lon0+360]
        lon0 = self.lon0
        lon1 = rectify_longitude(lon1, self.lon0)

        dist2 = ((LON-lon1)*self.coslat)**2 + (LAT-lat1)**2

        # exclude points from mask
        if mask is not None:
            dist2[mask] = np.inf

        #dist = distance_on_unit_sphere(lat1,lon1,LAT,LON)
        ii = np.argmin(dist2) # return index for the flat array
        ix, jx = np.unravel_index(ii, np.shape(LON)) # transform into sub-indices i and j
        lon_s, lat_s =LON[ix, jx], LAT[ix, jx] # lon/lat of the closest point

        dist = distance_on_unit_sphere(lat1,lon1,lat_s,lon_s) # distance to the closest point
        #dist = dist[ix,jx]
        if maxdist is not None and dist > maxdist:
            print self
            raise ValueError('{}N, {}E: {:.0f} km from next grid point ({}N, {}E) (max: {:.0f} km)'.format(lat1,lon1,dist,lat_s,lon_s, maxdist))

        if mask is not None and np.any(mask[ix,jx]):
            raise Exception('hey, some indices fall in the mask !')

        return ix, jx

    def locate_points(self, pt_x, pt_y, maxdist=None, mask=None, raise_error = True):
        """ general method, can handle a list of points
        """
        if np.size(pt_x) == 1:
            return self.locate_point(pt_x, pt_y, maxdist=maxdist, mask=mask)
        else:
            ii = np.zeros_like(pt_x, dtype=int)
            jj = np.zeros_like(pt_x, dtype=int)
            for k, coords in enumerate(zip(pt_x, pt_y)):
                x, y = coords
                i,j = self.locate_point(x, y, maxdist=maxdist, mask=mask)
                    
                ii[k] = i
                jj[k] = j
            return ii, jj

    def __repr__(self):
        return "{},{}N & {},{}E (lon0:{}E)".format(self.y.min(),self.y.max(),self.x.min(),self.x.max(),self.lon0)


def find_indices_coordinates(lat, lon, lats, lons, lons_lims = None, mask = None, search_radius = 300):
    """ Find grid indices corresponding to required coordinates
 
    DEPRECATED, for back compatibility only

    If mask is provided (boolean grid) return results for which mask is True
    """
    return LonLatGrid(lons, lats, lon0 = lons_lims[0]).locate_point(lon, lat, mask=mask, maxdist=search_radius)


def rectify_longitude(lon, lon0=0, sort=True):
    """ change the longitude to a particular reference 

    input:
        lon
        lon0 [default 0]
    output:
        lon: longitude on the domain [lon0, lon0+360]
    """
    # scalar
    if np.ndim(lon) == 0:
        if lon < lon0: lon += 360
        if lon > lon0+360: lon -= 360

    # numpy array
    else:
        lon = lon.copy()
        lon[lon<lon0] += 360
        lon[lon>lon0+360] -= 360

        # Re-sort the longitude axis
        if sort:
            lonf = lon.flatten()
            ii = np.argsort(lonf)
            lon[:] = lonf[ii].reshape(lon.shape)

    return lon


def rectify_longitude_data(lon, data, lon0):
    """ change the longitude to a particular reference 

    input:
        lon
        lon0 [default 0]
    output:
        lon: longitude on the domain [lon0, lon0+360]
    """
    assert lon.ndim == 1, "must be 1-D lon data"
    assert data.ndim in [1,2], "must be 1-D or 2-D data"

    lon = lon.copy()
    lon[lon<lon0] += 360
    lon[lon>lon0+360] -= 360
    ii = np.argsort(lon)
    lon = lon[ii]

    if data.ndim == 1:
        data = data[ii]
    else:
        data = data[:,ii]

    return lon, data

def distance_on_unit_sphere(lat1, long1, lat2, long2):
    """
    Modified after: http://www.johndcook.com/python_longitude_latitude.html
    """

    earth_radius_km = 6373.

    if np.any(np.isnan(lat1)): raise Exception('NaN in the data!')
    if np.any(np.isnan(lat2)): raise Exception('NaN in the data!')
    if np.any(np.isnan(long1)): raise Exception('NaN in the data!')
    if np.any(np.isnan(long2)): raise Exception('NaN in the data!')

    # make sure longitude are between 0 and 360
    long1 = rectify_longitude(long1, 0)
    long2 = rectify_longitude(long2, 0)

    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) = 
    #  sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) + 
        np.cos(phi1)*np.cos(phi2))
    arc = np.arccos( np.round(cos,6) )  # numerical error can make cos > 1 ==> need to round

    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.

    dist_km = arc * earth_radius_km

    if np.any(np.isnan(dist_km)): raise Exception('NaN in the calculated distance!')
    return dist_km

def distance(lon1, lat1, lon2, lat2):
    """ compute distance between two points of the Earth surface, in km
    """
    r = 6371 # Earth radius in km
    dist1 = np.sqrt(((lon1-lon2)*cos(0.5*(lat1+lat2)/180.*np.pi))**2 + (lat1-lat2)**2)
    return dist1/180.*np.pi*r # convert to radian and multiply by Earth Radius to get the distance



def interp2(lon, lat, data, lon2, lat2, order=1):
    """ Equivalent to matlab interp2

    order:
    - 0 : nearest neighbout
    - 1 : linear (default)
    """
    from dimarray.compat.basemap import interp
    if np.ndim(lon2) == 1:
        lon2, lat2 = np.meshgrid(lon2, lat2)
    return  interp(data, lon, lat, lon2, lat2, order=order)

def interpolate_1x1(lon, lat, data, lon0=0.):
    """ interpolate on a standard grid
    """
    # already standard
    if is_standard_grid(lon, lat, lon0=lon0):
        return lon, lat, data

    # 
    lon, data = rectify_longitude_data(lon, data, lon0)

    res = 1
    nx = int(360/res)
    ny = int(180/res)

    lon360 = np.linspace(lon0+0.5,lon0+359.5,nx)
    lat180 = np.linspace(-89.5,89.5,ny)

    print "Interpolate onto a standard 1deg grid...",
    if np.ndim(data) == 2:
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
        data = interp2(lon, lat, data, lon360,lat180, order=0)
        lon, lat = lon360, lat180

    elif np.ndim(data) == 3:
        nt = np.size(data,0)
        data180x360 = np.zeros((nt,ny,nx))
        for i in range(nt):
            data180x360[i,:,:] = interp2(lon, lat, np.squeeze(data[i,:,:]), lon360,lat180)
        lon, lat, data = lon360, lat180, data180x360
    print "Done"

    return lon, lat, data

def is_standard_grid(lon, lat, lon0=0.):
    """ True if standard grid: 1deg, 0,360, -90,90
    """
    if np.size(lon) == 360 and np.size(lat) == 180 \
            and lon[0] == lon0+0.5 and lat[0] == -79.5:
        return True
    else:
        return False

#
# Keep UNUSED code here for reference...
#
class KDTreeGrid(object):
    """ Class representing a grid
    with tools for fast nearest neihbor searching

    Thanks to:
    http://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
    """
    def __init__(self, x, y):
        """ initialize the grid 
        x, y: input array (1-D or 2-D), assume that the data it refers to 
        has y as first dimension and x as second (cartesian coordinate convention)
        """
        self.x = x
        self.y = y
        self._KDTree = None # for fast index searching

    def locate_points(self, pt_x, pt_y, maxdist=None):
        """ Find indices corresponding to a given point

        Input:
            pt_x, pt_y: floats or arrays
            maxdist : maximum distance allowed (computed as x**2 + y**2 !)
        Outputs:
            i, j      : y and x indices (cartesian convention)
        """
        if self._KDTree is None:
            self._build_KDTree()

        pt = self._transform(pt_x, pt_y)

        # derive linear index
        dist, ind = self._KDTree.query(pt)

        # transform into sub-indices
        i, j = np.unravel_index(ind, self.shape)

        if maxdist is not None and dist > maxdist:
            raise Exception('{:.0f} km from next grid point (max: {:.0f} km)'.format(dist, maxdist))

        # "flatten"
        if np.size(pt_x) == 1:
            i = int(i)
            j = int(j)

        return i, j

    def _build_KDTree(self):
        """ Build a KDTree with cKDTree from scipy.spatial
        """
        from scipy.spatial import cKDTree
        x, y = self.x, self.y

        # use 2-D arrays
        if np.ndim(x) == 1:
            x, y = np.meshgrid(x, y)

        self.shape = np.shape(x)

        combined_x_y_arrays = self._transform(x, y)

        self._KDTree = cKDTree(combined_x_y_arrays)

    @staticmethod
    def _transform(x, y):
        """ transform the coordinates to have an appropriate dimension
        """
        x = np.array(x)
        y = np.array(y)
        return np.dstack([x.ravel(), y.ravel()])[0]

class KDTreeLonLatGrid(KDTreeGrid):
    """ implementation of Grid assuming x and y are lon/lat, only to have the maxdist in km
    (but has no influence on the search algo...)
    """

    def locate_points(self, pt_x, pt_y, maxdist=None):
        """ Include additional check with distance on a lon/lat sphere
        """
        i, j = super(LonLatGrid_fast, self).locate_points(pt_x, pt_y, maxdist=None)

        if maxdist is None:
            return i,j

        # corresponding lon/lat point(s) on the grid
        if np.ndim(self.x) == 1:
            lon = self.x[j]
            lat = self.y[i]
        else:
            lon = self.x[i,j]
            lat = self.y[i,j]

        # compute distance(s) between nearest neighbor(s) and original point(s)
        if np.size(pt_x) == 1:
            dist = distance_on_unit_sphere(pt_y, pt_x, lat, lon)
        else:
            dist = 0.
            for k, v in enumerate(pt_x):
                kdist = distance_on_unit_sphere(pt_y[k], pt_x[k], lat[k], lon[k])
                dist = max(dist, kdist)

        # check against max tolerated distance
        if dist > maxdist:
            raise Exception('{:.0f} km from next grid point (max: {:.0f} km)'.format(dist, maxdist))

        return i,j

#
# Deprecated code 
#

# def find_indices_coordinates(lat, lon, lats, lons, lons_lims = None, mask = None, search_radius = 300):
#     """ Find grid indices corresponding to required coordinates
# 
#     If mask is provided (boolean grid) return results for which mask is True
#     """
#     # proceed to offset -180-180 / 0-360 offset
#     if lons_lims is None:
#         lons_lims = [0.,360.]
#     if lon < lons_lims[0]:
#         lon = lon + 360
#     if lon > lons_lims[1]:
#         lon = lon - 360
# 
#     # Nearest indices
#     i = np.argmin(np.abs(lats-lat))
#     j = np.argmin(np.abs(lons-lon))
# 
#     # Need to search 2d if mask if provided
#     if mask is not None:
#         lon2d, lat2d = np.meshgrid(lons, lats)
#         dist = distance_on_unit_sphere(lat2d[mask], lon2d[mask], lat, lon)
#         imask = np.argmin(dist) # linear index on the mask
# 
#         # the long way (it could be optimized by converting ii in i,j...):
#         ij = np.where(mask.flatten())[0][imask]        
#         lon_found = lon2d.flatten()[ij]
#         lat_found = lat2d.flatten()[ij]
#         i = np.argmin(np.abs(lats-lat_found))
#         j = np.argmin(np.abs(lons-lon_found))
# 
#         if  distance_on_unit_sphere(lats[i],lons[j],lat,lon) > search_radius:
#             #print 'required:',lat, lon
#             #print 'found:',lats[i], lons[j]
#             print 'warning: failed to find the required coordinates within the search radius of',search_radius,'km'
#             mask = None
# 
#     # Normal algo:
#     if mask is None:
# 
#         if  distance_on_unit_sphere(lats[i],lons[j],lat,lon) > search_radius:
#             print 'required:',lat, lon
#             print 'found:',lats[i], lons[j]
#             raise Exception('Failed to find the required coordinates within the search radius')
# 
#     return i,j
# 
# def find_indices_coordinates_fast():
#     """
# 
#     """
#     # Search center
#     [mj,j] = min(abs(lon-lon_grd));
#     [mi,i] = min(abs(lat-lat_grd));
# 
#     MSK = zeros[Ni, Nj];
#     DIST = ones[Ni, Nj]*99999; # distance between points and the center
# 
#     # ==============================================================
#     # Consider several cases based on input arguments rad and domain
#     # ==============================================================
# 
#     # Search domain = distance && domain of definition
#     if rad > 0:
#         # Find all points within the search domain 
#         nearest = 'n';
#     else:
#         # Only look at the nearest point within the search domain 
#         nearest = 'y';
#         rad = abs(rad);
#     if rad == 0: # do not worry about a radius, just take the closed grid point
#         MSK[i,j] = 1;
#         DIST[i,j] = calcdist(lat,lon,lat_grd[i],lon_grd[j]);
#     else:
#         # look in a disk of radius rad
#         # estimate the number of grid steps to explore
#         # ...look at grid step in km
#         dx = (lon_grd[j+1]-lon_grd[j])*cos(lat)*111;
#         dy = (lat_grd[i+1]-lat_grd[i])*111;
#         # ...deduce an exploration domain (security factor 2)
#         di = max(round(rad/dy*2),10);
#         dj = max(round(rad/dx*2),10);
#         for ii in range(i-di,i+di+1):
#             for jj in range(j-dj,j+dj+1):
#                 if jj <= 0:
#                     jj = len(lon_grd) + jj;
#                 if jj > len(lon_grd):
#                     jj = jj - len(lon_grd); 
#                 if ii > len(lat_grd) or ii <= 0:
#                     continue
#                 dist = calcdist(lat,lon,lat_grd[ii],lon_grd[jj]);
#                 if dist < rad and domain[ii,jj]:
#                     MSK[ii,jj] = 1;
#                     DIST[ii,jj] = dist;
# 
#         # Now select the points
#         indices = find[MSK];
#         if isempty(indices):
#             warning('No valid point found, return closest point.')
#             MSK[i,j] = 1;
#         else:
#             # Find the nearest point
#             if nearest == 'y':
#                 [m,indmin] = np.min(DIST);
#                 MSK = false[Ni, Nj];
#                 MSK[indmin] = true;
#                 if DIST[MSK] != m or DIST[MSK] > rad:
#                     DIST[MSK],m,indmin,rad
#                     error('Problem somewhere !')
#             else:
#                 # Return all point within the prescribed distance
#                 pass
