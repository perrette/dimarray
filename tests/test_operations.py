""" test operations with dimarrays
"""
import pytest
import dimarray as da
import numpy as np
from numpy.testing import assert_approx_equal as assert_aeq, assert_equal as assert_eq

#import doctest
#from dimarray.testing import get_globals, testfile as _testfile, testmod as _testmod # for doctest

@pytest.fixture
def simple_dimarray(shape=(2,3), dims=('x0', 'x1'), axes=None, dtype=float):
    a = np.arange(np.prod(shape)).reshape(shape)
    return da.DimArray(a, dims=dims, axes=axes, dtype=dtype)

def test_equal(simple_dimarray):
    a = simple_dimarray
    assert np.all(a == a)
    assert np.all(a != a + 1)
    assert a != a.ix[0]

def test_operations(simple_dimarray):
    """
    Operations
    """
    a = simple_dimarray
    
    # scalar operations
    assert np.all(a + 2 == a.values + 2)
    assert np.all(a / 2 == a.values / 2)
    assert np.all(a // 2 == a.values // 2)  # floor division
    assert np.all(a * 2 == a.values * 2)  
    assert np.all(a ** 2 == a.values ** 2)  
    #assert_equal(a + 2, a.values + 2)
    #assert_equal(a / 2, a.values / 2)
    #assert_equal(a // 2, a.values // 2)  # floor division
    #assert_equal(a * 2, a.values * 2)  
    #assert_equal(a ** 2, a.values ** 2)  

    # operations with dimarrays
    a += 2
    assert np.all(a + a == a.values + a.values)
    assert np.all(a - a == a.values - a.values)
    assert np.all(a / a == np.true_divide(a.values, a.values))
    assert np.all(a // a == a.values // a.values)
    assert np.all(a ** a == a.values ** a.values)
    assert np.all((a == a) == (a.values == a.values))
    assert np.all((a >= a) == (a.values >= a.values))
    assert np.all((a > a) == (a.values > a.values))
    assert np.all((a <= a) == (a.values <= a.values))
    assert np.all((a < a) == (a.values < a.values))

    # misc
    assert np.all(a+2 == a + np.ones(a.shape)*2)
    assert np.all(a+a == a*2)
    assert np.all(a*a == a**2)
    assert np.all((a - a.values) == a - a)

