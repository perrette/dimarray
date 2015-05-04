""" Test suite for axes objects """
import pytest
import numpy as np
from dimarray import Axis, Axes
from dimarray.testing import assert_equal_axes

@pytest.fixture
def ax0():
    return Axis(['a','b'], 'd0')

@pytest.fixture
def ax1():
    return Axis([11., 22., 33.], 'd1')

@pytest.fixture
def axes(ax0, ax1):
    return Axes([ax0, ax1])

def test_copy(axes):
    axes_n = axes.copy()
    assert axes == axes_n

    # copy applies at least on the container (list)
    ax2 = Axis([111., 222., 333.], 'd2')
    axes_n[1] = ax2
    assert axes != axes_n

    # deepcopy also applies recursively on the axes
    # NOTE: this may change in the future to stay
    # close to python's list behaviour.
    axes_n = axes.copy()
    axes_n[0][0] = 'c'
    assert axes != axes_n

def test_append(ax0, ax1):
    # test copy/ref behaviour when appending an object
    axes = Axes()
    axes.append(ax0)
    assert ax0 is axes[0]
    axes.append(ax1)
    assert ax1 is axes[1]

@pytest.fixture(params=['f','i','O'])
def dtype(request):
    return np.dtype(request.param)

def test_merge_increasing(dtype):
    ax0 = Axis([1,2,3], 'd0', dtype=dtype)
    ax1 = Axis([2,3,4], 'd0', dtype=dtype)

    ax2 = ax0.union(ax1)
    expected = Axis([1,2,3,4], 'd0', dtype=dtype)
    assert_equal_axes(expected, ax2)

    ax3 = ax0.intersection(ax1)
    expected = Axis([2,3], 'd0', dtype=dtype)
    assert_equal_axes(expected, ax3)

def test_merge_decreasing(dtype):
    ax0 = Axis([3,2,1], 'd0', dtype=dtype)
    ax1 = Axis([4,3,2], 'd0', dtype=dtype)

    ax2 = ax0.union(ax1)
    expected = Axis([4,3,2,1], 'd0', dtype=dtype)
    assert_equal_axes(expected, ax2)

    ax3 = ax0.intersection(ax1)
    expected = Axis([3,2], 'd0', dtype=dtype)
    assert_equal_axes(expected, ax3)

def test_merge_chaos(dtype):
    ax0 = Axis([1,2,4], 'd0', dtype=dtype)
    ax1 = Axis([3,4,1], 'd0', dtype=dtype)

    ax2 = ax0.union(ax1)
    # input axes are not sorted, do nothing
    expected = Axis([1,2,4,3], 'd0', dtype=dtype)
    assert_equal_axes(expected, ax2)

    ax3 = ax0.intersection(ax1)
    expected = Axis([1,4], 'd0', dtype=dtype)
    assert_equal_axes(expected, ax3)

def test_merge_mixed_type():
    ax0 = Axis(['a','b',33,11], 'd0', dtype='O')
    ax1 = Axis([11., 33., 22.], 'd0')

    ax2 = ax0.union(ax1)
    expected = Axis(['a','b', 33, 11, 22.], 'd0', dtype='O')
    assert_equal_axes(expected, ax2)

    ax3 = ax0.intersection(ax1)
    expected = Axis([33, 11], 'd0', dtype='O')
    assert_equal_axes(expected, ax3)
