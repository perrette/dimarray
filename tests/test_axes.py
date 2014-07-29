""" Test suite for axes objects """
import pytest
import numpy as np
from dimarray import Axis, Axes

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

