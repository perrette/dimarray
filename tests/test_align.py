from __future__ import print_function
import numpy as np
from dimarray import DimArray
from dimarray.testing import assert_equal_dimarrays
from dimarray import align, stack, concatenate, Dataset, stack_ds, concatenate_ds
import pytest

def _make_datasets(*arrays):
    # extend the tests to datasets with one array
    return [Dataset({'a':a}) for a in arrays]

def test_align():
    " test slices in an increasing axis"
    a = DimArray([1, 2, 3, 4], axes=[[1, 2, 3, 4]], dims=['dim0'])
    b = DimArray([0, 2, 4, 6], axes=[[0, 2, 4, 6]], dims=['dim0'])

    # outer join
    # expected
    a2 = DimArray([np.nan, 1, 2, 3, 4, np.nan], axes=[[0, 1, 2, 3, 4, 6]], dims=['dim0'])
    b2 = DimArray([0, np.nan, 2, np.nan, 4, 6], axes=[[0, 1, 2, 3, 4, 6]], dims=['dim0'])
    # got
    a2_got, b2_got = align([a, b], join="outer")
    # check
    assert_equal_dimarrays(a2, a2_got)
    assert_equal_dimarrays(b2, b2_got)

    # inner join
    # expected
    a3 = DimArray([2, 4], axes=[[2,4]], dims=['dim0'])
    b3 = DimArray([2, 4], axes=[[2,4]], dims=['dim0'])
    # got
    a3_got, b3_got = align([a, b], join="inner")
    # check
    assert_equal_dimarrays(a3, a3_got)
    assert_equal_dimarrays(b3, b3_got)


def test_align_unsorted():

    a_sorted = DimArray([1, 2, 3, 4], axes=[[1, 2, 3, 4]], dims=['dim0'])
    a_mess = DimArray([3, 1, 4, 2], axes=[[3, 1, 4, 2]], dims=['dim0'])
    b_sorted = DimArray([11, 22, np.nan, 44], axes=[[1, 2, 3, 4]], dims=['dim0'])
    b_mess = DimArray([22, 44, 11], axes=[[2, 4, 1]], dims=['dim0'])

    # second array unsorted
    a_got, b_got = align([a_sorted, b_mess], join="outer")

    assert_equal_dimarrays(a_got, a_sorted)
    assert_equal_dimarrays(b_got, b_sorted)

    # first array unsorted
    b_got, a_got = align([b_mess, a_sorted], join="outer")

    assert not np.all(a_got.dim0 == a_sorted.dim0)  # not equal because not ordered

    a_got = a_got.sort_axis()
    b_got = b_got.sort_axis()

    assert_equal_dimarrays(a_got, a_sorted)
    assert_equal_dimarrays(b_got, b_sorted)

    # do the same, but pass as command line
    a_got, b_got = align([a_sorted, b_mess], join="outer", sort=True)

    # two arrays unsorted
    a_got, b_got = align([a_mess, b_mess], join="outer", sort=True)

    assert_equal_dimarrays(a_got, a_sorted)
    assert_equal_dimarrays(b_got, b_sorted)

def test_stack():

    a = DimArray([1,2,3], dims=['x0'])
    b = DimArray([11,22,33], dims=['x0'])
    c = DimArray([[ 1,  2,  3],
                  [11, 22, 33]], axes=[['a', 'b'], [0, 1, 2]], dims=['stackdim', 'x0'])

    c_got = stack([a, b], axis='stackdim', keys=['a','b'])
    c_got_ds = stack_ds(_make_datasets(a, b), axis='stackdim', keys=['a','b'])

    assert_equal_dimarrays(c_got, c)
    assert_equal_dimarrays(c_got_ds['a'], c)

def test_stack_align():
    a = DimArray([1,2,3], axes=[[0,1,2]], dims=['x0'])
    b = DimArray([33,11], axes=[[2,0]], dims=['x0'])

    c_got = stack([b, a], axis='stackdim', align=True, sort=True, keys=['a','b'])
    c_got_ds = stack_ds(_make_datasets(b, a), axis='stackdim', align=True, sort=True, keys=['a','b'])

    c = DimArray([[11., np.nan, 33.],
                  [ 1.,     2.,  3.]], axes=[['a', 'b'], [0, 1, 2]], dims=['stackdim', 'x0'])

    assert_equal_dimarrays(c_got, c)
    assert_equal_dimarrays(c_got_ds['a'], c)

def test_stack_fails():
    # Should use concatenate instead, because axis is not new !
    a = DimArray([1,2,3], dims=['x0'])
    b = DimArray([11,22,33], dims=['x0'])

    with pytest.raises(ValueError):
        c_got = stack([a, b], axis='x0')

def test_concatenate_1d():
    a = DimArray([1,2,3], dims=['x0'])
    b = DimArray([11,22,33], dims=['x0'])
    c = DimArray([1,  2,  3, 11, 22, 33], axes=[[0, 1, 2, 0, 1, 2]], dims=['x0'])
    # a = DimArray([1,2,3], axes=[['a','b','c']])
    # b = DimArray([4,5,6], axes=[['d','e','f']])
    # c = DimArray([1, 2, 3, 4, 5, 6], axes=[['a','b','c','d','e','f']])

    c_got = concatenate((a, b))
    c_got_ds = concatenate_ds(_make_datasets(a, b))

    assert_equal_dimarrays(c_got, c)
    assert_equal_dimarrays(c_got_ds['a'], c)

def test_concatenate_2d():

    a = DimArray([[ 1,  2,  3],
                  [11, 22, 33]], axes=[[0,1],[2,1,0]])

    b = DimArray([[44, 55,  66],
                  [4,   5,   6]], axes=[[1,0],[2,1,0]])

    c0_got = concatenate((a, b), axis=0)
    c0_got_ds = concatenate_ds(_make_datasets(a, b), axis=0)

    c0 = DimArray([[ 1,  2,  3],
                   [11, 22, 33],  
                   [44, 55, 66],  
                   [4,   5,  6]], axes=[[0,1,1,0],[2,1,0]]) 

    assert_equal_dimarrays(c0_got, c0)
    assert_equal_dimarrays(c0_got_ds['a'], c0)
                  
    # axis "x0" is not aligned !
    with pytest.raises(ValueError):
        c1_got = concatenate((a, b), axis=1)
        print(c1_got)

    c1_got = concatenate((a, b), axis=1, align=True, sort=True)
    c1_got_ds = concatenate_ds(_make_datasets(a, b), axis=1, align=True, sort=True)

    c1 = DimArray([[ 1,  2,  3,  4,  5,  6],
                   [11, 22, 33, 44, 55, 66]], axes=[[0,1],[2,1,0,2,1,0]])

    assert_equal_dimarrays(c1_got, c1)
    assert_equal_dimarrays(c1_got_ds['a'], c1)
    
