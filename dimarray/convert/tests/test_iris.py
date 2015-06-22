#import unittest
import pytest

try:
    import iris.util
except:
    pytestmark = pytest.mark.skipif(True, reason="iris is not installed or too old (iris.itil cannot be imported")

from dimarray.geo import GeoArray
from numpy.testing import assert_allclose

@pytest.fixture
def geo_array():
    a = GeoArray([[1.,2.,3.],[4.,5.,6.]], lat=[30., 70.], lon=[-50., 30., 110.])
    a.units = 'meter' 
    a.name = 'myarray'
    a.long_name = 'test array for iris conversion'
    a.grid_mapping = dict(
            grid_mapping_name = 'longitude_latitude')
    return a

def test_to_cube(geo_array):
    geo_array.to_cube()

def test_two_ways(geo_array):
    #da.GeoArray.from_cube(grl['surfvelx'].squeeze().to_cube())
    cube = geo_array.to_cube()
    a = GeoArray.from_cube(cube)
    assert_allclose(a, geo_array)
    # assert a._metadata == geo_array._metadata # will fail
