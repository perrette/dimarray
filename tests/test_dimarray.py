""" Test basic dimarray behaviour (copy, attributes)
"""
import pytest
import numpy as np
from dimarray import DimArray, Axis

@pytest.fixture
def ax0():
    x0 = ['a','b']
    return Axis(x0,'d0')

@pytest.fixture
def ax1():
    x1 = [11., 22., 33.]
    return Axis(x1,'d1')

@pytest.fixture
def a(ax0, ax1):
    values = np.random.randn(ax0.size, ax1.size)
    a = DimArray(values, axes=[ax0, ax1])
    a.units = 'meters'
    a.mutable_att = ['a1','a2']
    return a

def test_copy(a):
    b = a.copy()
    assert (a == b).all()
    assert a is not b

    # deep copy by default for array values
    b['a',11] = 55.
    assert not np.all(a == b)

    # axes attribute
    assert a.axes == b.axes
    assert a.axes is not b.axes

    # each individual axis is equal
    assert np.all([a.axes[i] == b.axes[i] for i in range(a.ndim)])

    # but does not occupy the same space in memory
    assert not np.any([a.axes[i] is b.axes[i] for i in range(a.ndim)])

    # metadata are also copied
    assert a.units == b.units 
    assert a.mutable_att == b.mutable_att 
    
    b.units = 'millimeters'
    assert a.units != b.units

    # ... including mutable attributes
    b.mutable_att[0] = 'new_att'
    assert a.mutable_att != b.mutable_att

def test_dims(a):
    """ update dimensions
    """
    assert a.dims == ('d0','d1')
    a.dims = ('newa','newb')
    assert a.dims == ('newa','newb')
    assert a.axes[0].name == 'newa'
    assert a.axes[1].name == 'newb'

def test_labels(a, ax0, ax1):
    """ update labels
    """
    assert a.labels == (ax0.values, ax1.values)
    new0 = ['c','d']
    new1 = [0.,0.,0.]
    a.labels = [ new0, new1 ]
    assert np.all(a.axes['d0'].values == new0)
    assert np.all(a.axes['d1'].values == new1)

#     >>> a = DimArray(axes=[[1,2,3]], dims=['x0'])
#     >>> a.x0 
#     array([1, 2, 3])
#     >>> a.x0 = a.x0*2
#     >>> a.x0
#     array([2, 4, 6])
#     >>> a.x0 = a.x0*1.  # conversion to float
#     >>> a.x0
#     array([ 2.,  4.,  6.])
#     >>> a.x0 = list('abc')  # or any other type
#     >>> a.x0
#     array(['a', 'b', 'c'], dtype=object)
#     """
