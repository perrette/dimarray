""" Basic unit tests for a GeoArray
""" 
import pytest
import numpy as np
from numpy.random import randn
from dimarray.geo import GeoArray, ones
from dimarray import Dataset

@pytest.fixture(params=[(3,2)])
def shape(request):
    return request.param

@pytest.fixture(params=[(3,2)])
def a2d(request):
    shape = request.param
    ni, nj = shape
    lat = np.linspace(-89.5, 89.5, ni)
    lon = np.linspace(-179.9, 179.5, nj)
    return GeoArray(randn(*shape), lon=lon, lat=lat)

# test type
def test_type(a2d):
    """ test that the GeoArray type is conserved
    """
    assert type(a2d*2) is GeoArray
    assert type(a2d+a2d) is GeoArray
    assert type(a2d.values+a2d) is GeoArray
    assert type(2**a2d) is GeoArray

    assert type(a2d.mean(axis=0)) is GeoArray
    assert type(a2d.sum(axis=0)) is GeoArray

    ds = Dataset()
    ds['a'] = a2d
    assert type(ds['a']) is GeoArray

    ds = Dataset({'a':a2d,'b':a2d*2})
    assert type(ds['a']) is GeoArray


# transformations
def test_weighted_mean(a2d):
    values = a2d.values
    weights = np.cos(np.radians(a2d.lat))
    weights, values = np.broadcast_arrays(weights, values.T) # align dimensions
    res0 = np.sum(values * weights) / np.sum(weights)
    res1 = a2d.mean()
    assert res0 == res1

