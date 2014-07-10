""" test CRS transforms
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose

import dimarray as da
from dimarray.geo.crs import get_grid_mapping, Proj4
from dimarray.geo import GeoArray

class TestCRS(unittest.TestCase):
    """ test netCDF formulation versus Proj.4
    """
    grid_mapping = None
    proj4_init = None

    def setUp(self):
        shape = 180, 360
        ni, nj = shape # full globe
        lat = np.linspace(-89.5, 89.5, ni)
        lon = np.linspace(-179.9, 179.5, nj)
        self.a = GeoArray(np.random.randn(*shape), lon=lon, lat=lat)

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

    def test_transform(self):
        pass
        

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

    def test_coords(self):
        ncfile = da.get_ncfile('greenland_velocity.nc')
        try:
            ds = da.read_nc(ncfile, ['surfvelmag','mapping','lon','lat'])
        except:
            warnings.warn('could not test transform on real file')
            #return # no netCDF4?

        vmag, mapping, lon, lat = ds.values()
        mapping = mapping._metadata
        del mapping['name']  # should probably remove automatic naming after being in a dataset
        
        crs = get_grid_mapping(mapping)
        lonlat = get_grid_mapping('geodetic')
        x1, y1 = np.meshgrid(vmag.x1, vmag.y1)
        coords = lonlat.transform_points(crs, x1, y1)
        newlon, newlat, z = np.rollaxis(coords, -1) # bring the last axis at the front

        assert_allclose(newlon, lon)
        assert_allclose(newlat, lat)

class TestGeodetic(TestCRS):

    grid_mapping = dict(
        grid_mapping_name = 'latitude_longitude', 
        ellipsoid = 'WGS84', 
        )  # CF-1.4

    proj4_init = '+ellps=WGS84 +proj=lonlat'




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
