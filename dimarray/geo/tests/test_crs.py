""" test CRS transforms
"""
from __future__ import division

import unittest
import numpy as np
from numpy.testing import assert_allclose
import dimarray as da
from dimarray.geo.crs import get_crs, Proj4
from dimarray.geo import GeoArray, transform, transform_vectors

class TestCRS(unittest.TestCase):
    """ test netCDF formulation versus Proj.4
    """
    grid_mapping = None
    proj4_init = None

    # used to test vectors transformations:

    def test_from_cf(self):
        if not self.grid_mapping: return # only subclassing
        crs = get_crs(self.grid_mapping)

        expected = self.proj4_init + ' +no_defs'
        self.assertEqual(crs.proj4_init, expected)

        ## just in case
        #expected = self.grid_mapping
        #self.assertEqual(crs.cf_params, expected)

# Test removed as support from from_proj4 was removed
#    def test_from_proj4(self):
#        if not self.grid_mapping: return # only subclassing
#        
#        # test CF_CRS to CF conversion with knowledge of grid_mapping
#        crs = get_crs(self.proj4_init)
#        expected = self.grid_mapping
#        self.assertEqual(crs.cf_params, expected)


class TestStereographic(TestCRS):

    grid_mapping = dict(
        grid_mapping_name = 'stereographic', 
        latitude_of_projection_origin = 80., 
        longitude_of_projection_origin = -39.,
        scale_factor_at_projection_origin = 0.9,
        false_easting = 0.,  # default
        false_northing = 0., 
        ellipsoid = 'WGS84', 
        )  # CF-1.4

    proj4_init = '+ellps=WGS84 +proj=stere +lat_0=80.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +k_0=0.9'

class TestPolarStereographic(TestCRS):

    grid_mapping = dict(
        grid_mapping_name = 'polar_stereographic', 
        latitude_of_projection_origin = 90., 
        straight_vertical_longitude_from_pole = -39.,
        standard_parallel = 71.,
        false_easting = 0.,  # default
        false_northing = 0., 
        ellipsoid = 'WGS84', 
        )  # CF-1.4

    proj4_init = '+ellps=WGS84 +proj=stere +lat_0=90.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +lat_ts=71.0'


class TestLatitudeLongitude(TestCRS):

    grid_mapping = dict(
        grid_mapping_name = 'latitude_longitude', 
        ellipsoid = 'WGS84', 
        longitude_of_prime_meridian = 0.,
        )  # CF-1.4

    proj4_init = '+ellps=WGS84 +proj=eqc +lon_0=0.0'

    _test_vector = False # not a plane system: does not work

class TestRotatedPoles(TestCRS):

    grid_mapping = dict(
            grid_mapping_name = 'rotated_latitude_longitude',
            grid_north_pole_longitude =0., 
            grid_north_pole_latitude =90.)

    proj4_init = '+ellps=WGS84 +proj=ob_tran +o_proj=latlon +o_lon_p=0 +o_lat_p=90.0 +lon_0=180.0 +to_meter=0.0174532925199'

    _test_vector = False # not a plane system: does not work


class TestTransverseMercator(TestCRS):

    grid_mapping = dict(
            grid_mapping_name = 'transverse_mercator',
            scale_factor_at_central_meridian = 1.,
            longitude_of_central_meridian = 0.,
            latitude_of_projection_origin = 0.,
            false_northing = 0.,
            false_easting = 0.,
            ellipsoid = 'WGS84', 
            )

    proj4_init = '+ellps=WGS84 +proj=tmerc +lon_0=0.0 +lat_0=0.0 +k=1.0 +x_0=0.0 +y_0=0.0 +units=m'

    _test_vector = False # buggy

    def test_project(self):
        pass # bug


class TestPolarStereographicData(unittest.TestCase):

    def setUp(self):
        ncfile = da.get_ncfile('greenland_velocity.nc')
        try:
            self.grl = da.read_nc(ncfile)
        except:
            self.grl = None
            warnings.warn('could not read netCDF: no test_coords')
            return

        grl = self.grl

        self.vmag = grl['surfvelmag']
        self.vx = grl['surfvelx']
        self.vy = grl['surfvely']
        self.lon = grl['lon']
        self.lat = grl['lat']
        self.mapping = grl['mapping']._metadata()

    def test_coords(self):
        if self.grl is None: return

        import cartopy.crs as ccrs
        vmag = self.vmag
        crs = get_crs(self.mapping)

        #lonlat = get_crs('geodetic')
        x1, y1 = np.meshgrid(vmag.x1, vmag.y1)
        pt_xyz = ccrs.Geodetic().transform_points(crs, x1, y1)
        newlon, newlat = pt_xyz[...,0], pt_xyz[..., 1]

        assert_allclose(newlon, self.lon)
        assert_allclose(newlat, self.lat)

    def test_transform(self):
        shape = 5,5
        ni, nj = shape # full globe
        x = np.linspace(-180, 180, nj)
        y = np.linspace(-85, 85, ni)
        a = GeoArray(np.random.randn(*shape), lat=y, lon=x)

        # basic transform
        at = transform(a, to_crs=self.mapping)

        # multi-dimensional
        a3d = a.newaxis('dummy_axis', [0, 1, 2])
        a3dt = transform(a3d, to_crs=self.mapping)

        expected = at
        assert_allclose(a3dt[0], expected)

    def test_vector(self):
        shape = 5,5
        ni, nj = shape # full globe

        # need some kind of plane projection, not long-lat for vectors
        crs = self.mapping.copy()
        crs['standard_parallel'] = 50
        crs['straight_vertical_longitude_from_pole'] = 70
        x = np.linspace(-5000e3, 5000e3, nj)
        y = np.linspace(-4000e3, 4000e3, ni)
        u = GeoArray(np.random.randn(*shape), y=y, x=x)
        v = GeoArray(np.random.randn(*shape), y=y, x=x)
        #u.mapping = crs
        #v.mapping = crs

        # basic transform
        ut,vt = transform_vectors(u,v, from_crs=crs, to_crs=self.mapping)

        # multi-dimensional
        u3d = u.newaxis('dummy_axis', [0, 1, 2])
        v3d = v.newaxis('dummy_axis', [0, 1, 2])
        u3dt, v3dt = transform_vectors(u3d,v3d, from_crs=crs, to_crs=self.mapping)

        assert_allclose(u3dt[0], ut)
        assert_allclose(v3dt[0], vt)
