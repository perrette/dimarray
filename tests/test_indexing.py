""" Test indexing
"""
import numpy as np
import pytest
from dimarray import DimArray
from dimarray.testing import assert_equal_dimarrays

@pytest.fixture
def v():
    return DimArray([[1,2],[3,4],[5,6],[7,8]], labels=[["a","b","c","d"], [10.,20.]], dims=['x0','x1'], dtype=float) 

def test_take(v):
    a = v[:,10]
    b = v.take(10, axis=1)
    c = v.take(10, axis='x1')
    d = v.take({'x1':10}) # dict
    e = v.take((slice(None),10)) # tuple

    assert np.all(a==b)
    assert np.all(a==c) 
    assert np.all(a==d) 
    assert np.all(a==e)

def test_nloc():
    # test nearest neighbor indexing
    a = DimArray([1,2,3,4], axes=[[2., 4., -1., 3.]])

    # scalar
    a.nloc[2] == a[2]
    a.nloc[2.4] == a[2]
    a.nloc[2.6] == a[3]
    a.nloc[10] == a[4]
    a.nloc[-10] == a[-1]

    # array
    assert_equal_dimarrays(a.nloc[[2.1]], a[[2]])

    # more dimensions
    b = a.newaxis('newdim', [10,20], pos=1)
    assert_equal_dimarrays(b.nloc[[2.1]], b[[2]])

def test_nloc_mixed():
    # test nearest neighbor indexing
    a = DimArray([[1,2],[3,4],[5,6],[7,8]], labels=[["a","b","c","d"], [10.,20.]], dims=['x0','x1'], dtype=float) 
    assert_equal_dimarrays(a.nloc[:,10.], a[:, 10.])
    assert_equal_dimarrays(a.nloc[:,5.], a[:, 10.])
    assert_equal_dimarrays(a.nloc[:,14.5], a[:, 10.])
    assert_equal_dimarrays(a.nloc[:,15.5], a[:, 20.])
    assert_equal_dimarrays(a.nloc['a'], a['a'])
    # assert_equal_dimarrays(a.nloc['e'], a['d'])

def test_slices_sorted():
    " test slices in an increasing axis"
    a = DimArray([1.5, 2.5, 3.5], axes=[[1.5, 2.5, 3.5]], dims=['dim0'])
    a0 = DimArray([2.5, 3.5], axes=[[2.5, 3.5]], dims=['dim0'])
    a1 = DimArray([2.5], axes=[[2.5]], dims=['dim0'])
    assert_equal_dimarrays(a[2.5:3.5], a0)
    assert_equal_dimarrays(a[2.5:4], a0)
    assert_equal_dimarrays(a[2:], a0)
    assert_equal_dimarrays(a[2:3], a1)
    assert_equal_dimarrays(a[3:2:-1], a1)

def test_slices_decreasing():
    " test slices in a decreasing axis"
    a = DimArray([3.5, 2.5, 1.5], axes=[[3.5, 2.5, 1.5]], dims=['dim0'])
    a0 = DimArray([3.5, 2.5], axes=[[3.5, 2.5]], dims=['dim0'])
    a1 = DimArray([2.5], axes=[[2.5]], dims=['dim0'])
    assert_equal_dimarrays(a[3.5:2.5], a0)
    assert_equal_dimarrays(a[2.5:3.5:-1], a0[::-1])
    assert_equal_dimarrays(a[4:2.5], a0)
    assert_equal_dimarrays(a[3:2], a1)

def test_slices_mixed():
    a = DimArray([2.5, 3.5, 1.5], axes=[[2.5, 3.5, 1.5]], dims=['dim0'])
    a0 = DimArray([3.5, 1.5], axes=[[3.5, 1.5]], dims=['dim0'])
    assert_equal_dimarrays(a[3.5:1.5], a0)
    assert_equal_dimarrays(a[1.5:3.5:-1], a0[::-1])

def test_slices_objects():
    a = DimArray(['c','a','b'], axes=[['c','a','b']], dims=['dim0'])
    a0 = DimArray(['c','a'], axes=[['c','a']], dims=['dim0'])
    assert_equal_dimarrays(a['c':'a'], a0)
