""" test CRS transforms
"""

import unittest

from dimarray.geo.crs import get_grid_mapping, _crs_to_cf_params, Proj4

#class TestCRS(unittest.TestCase):
#    """ test netCDF formulation versus Proj.4
#    """
#
#    def test_stereographic(self):
#
#        grid_mapping = dict(
#            grid_mapping_name = 'stereographic', 
#            latitude_of_projection_origin = 90., 
#            longitude_of_projection_origin = -39.,
#            scale_factor_at_projection_origin = 0.9,
#            false_easting = 0.,  # default
#            false_northing = 0., 
#            ellipsoid = 'WGS84', 
#            )  # CF-1.4
#
#        proj4_init = '+ellps=WGS84 +proj=stere +lat_0=90.0 +lon_0=-39.0 +x_0=0.0 +y_0=0.0 +k_0=0.9'
#
#        # test CF to CRS conversion
#        crs = get_grid_mapping(grid_mapping)
#        expected = proj4_init + ' +no_defs'
#        self.assertEqual(crs.proj4_init, expected)
#
#        # test CF_CRS to CF conversion with knowledge of grid_mapping
#        cf_params = _crs_to_cf_params(crs, crs.__class__)
#        self.assertEqual(cf_params, grid_mapping)
#
#        # same but need to look for actual class
#        cf_params = _crs_to_cf_params(crs)
#        self.assertEqual(cf_params, grid_mapping)
#
#        # test PROJ.4 to CF conversion
#        crs = _proj4_to_crs(proj4_init)
#        cf_params = _crs_to_cf_params(crs)
#        self.assertEqual(cf_params, grid_mapping)
#
#    def test_polar_stereographic(self):
#        grid_mapping = dict(
#            grid_mapping_name = 'polar_stereographic', 
#            latitude_of_projection_origin = 90., 
#            straight_vertical_longitude_from_pole = -39.,
#            standard_parallel = 71.,
#            false_easting = 0.,  # default
#            false_northing = 0., 
#            ellipsoid = 'WGS84', 
#            )  # CF-1.4

def test_stereographic():

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

    # test CF to CRS conversion
    crs = get_grid_mapping(grid_mapping)
    expected = proj4_init + ' +no_defs'
    assert crs.proj4_init == expected

    # test CF_CRS to CF conversion with knowledge of grid_mapping
    expected = grid_mapping
    assert  crs.cf_params == expected

    # test PROJ.4 to CF conversion
    crs = Proj4(proj4_init)
    assert crs.cf_params == expected
