""" test CRS transforms
"""
from __future__ import division

import unittest
import numpy as np
from numpy.testing import assert_allclose

import dimarray as da
from dimarray.geo.crs import get_grid_mapping, Proj4
from dimarray.geo import GeoArray, transform, transform_vectors

class TestCRS(unittest.TestCase):
    """ test netCDF formulation versus Proj.4
    """
    grid_mapping = None
    proj4_init = None

    #def setUp(self):
    #    shape = 180, 360
    #    ni, nj = shape # full globe
    #    lat = np.linspace(-89.5, 89.5, ni)
    #    lon = np.linspace(-179.9, 179.5, nj)
    #    self.a = GeoArray(np.random.randn(*shape), lon=lon, lat=lat)

    def test_from_cf(self):
        if not self.grid_mapping: return # only subclassing
        crs = get_grid_mapping(self.grid_mapping)

        expected = self.proj4_init + ' +no_defs'
        self.assertEqual(crs.proj4_init, expected)

        # just in case
        expected = self.grid_mapping
        self.assertEqual(crs.cf_params, expected)

    def test_from_proj4(self):
        if not self.grid_mapping: return # only subclassing
        
        # test CF_CRS to CF conversion with knowledge of grid_mapping
        crs = get_grid_mapping(self.proj4_init)
        expected = self.grid_mapping
        self.assertEqual(crs.cf_params, expected)


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
        self.mapping = grl['mapping']._metadata
        del self.mapping['name'] # should probably remove automatic naming after being in a dataset

    def test_coords(self):
        if self.grl is None: return

        vmag = self.vmag
        crs = get_grid_mapping(self.mapping)

        lonlat = get_grid_mapping('geodetic')
        x1, y1 = np.meshgrid(vmag.x1, vmag.y1)
        coords = lonlat.transform_points(crs, x1, y1)
        newlon, newlat, z = np.rollaxis(coords, -1) # bring the last axis at the front

        assert_allclose(newlon, self.lon)
        assert_allclose(newlat, self.lat)

    def test_transform(self):
        " just run a transform "
        if self.grl is None: return
        
        vmagt = transform(self.vmag, from_grid_mapping=self.mapping, \
                to_grid_mapping='geodetic', \
                xt=np.linspace(-75,-10,300), yt=np.linspace(60, 84, 300))

        self.assertGreater((~np.isnan(vmagt)).sum() , (~np.isnan(self.vmag)).sum()*0.25) # against all nan
        #assert_allclose((ut**2+vt**2, vmagt**2)

        ut, vt = transform_vectors(self.vx, self.vy, from_grid_mapping=self.mapping, \
                to_grid_mapping='geodetic', \
                xt=np.linspace(-75,-10,300), yt=np.linspace(60, 84, 300))

        #assert_allclose((ut**2+vt**2)**0.5, vmagt, atol=1e-2, rtol=0.01)
        # fraction of point close within 1%
        frac = np.isclose((ut**2+vt**2)**0.5, vmagt, rtol=0.01, equal_nan=True).sum() / vmagt.size
        self.assertGreaterEqual(frac, 0.7)  # multiplication + interp ==> does not work so well ??


class TestGeodetic(TestCRS):

    grid_mapping = dict(
        grid_mapping_name = 'latitude_longitude', 
        ellipsoid = 'WGS84', 
        )  # CF-1.4

    proj4_init = '+ellps=WGS84 +proj=lonlat'

class TestRotatedPoles(TestCRS):

    grid_mapping = dict(
            grid_mapping_name = 'rotated_latitude_longitude',
            grid_north_pole_longitude =0., 
            grid_north_pole_latitude =90.)

    proj4_init = '+ellps=WGS84 +proj=ob_tran +o_proj=latlon +o_lon_p=0 +o_lat_p=90.0 +lon_0=180.0 +to_meter=0.0174532925199'




  #  def test_polar_stereographic(self):
  #      grid_mapping = dict(
  #          grid_mapping_name = 'polar_stereographic', 
  #          latitude_of_projection_origin = 90., 
  #          straight_vertical_longitude_from_pole = -39.,
  #          standard_parallel = 71.,
  #          false_easting = 0.,  # default
  #          false_northing = 0., 
  #          ellipsoid = 'WGS84', 
  #          )  # CF-1.4

#def test_stereographic():
#
#    grid_mapping = dict(
#        grid_mapping_name = 'stereographic', 
#        latitude_of_projection_origin = 80., 
#        longitude_of_projection_origin = -39.,
#        scale_factor_at_projection_origin = 0.9,
#        false_easting = 0.,  # default
#        false_northing = 0., 
#        ellipsoid = 'WGS84', 
#        )  # CF-1.4
#
#    proj4_init = '+ellps=WGS84 +proj=stere +lat_0=80.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +k_0=0.9'
#
#    # test CF to CRS conversion
#    crs = get_grid_mapping(grid_mapping)
#    expected = proj4_init + ' +no_defs'
#    assert crs.proj4_init == expected
#
#    # test CF_CRS to CF conversion with knowledge of grid_mapping
#    expected = grid_mapping
#    assert  crs.cf_params == expected
#
#    # test PROJ.4 to CF conversion
#    crs = Proj4(proj4_init)
#    assert crs.cf_params == expected
