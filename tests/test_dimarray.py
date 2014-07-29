""" Test basic dimarray behaviour (copy, attributes)
"""
import pytest
import numpy as np
from dimarray import DimArray

@pytest.fixture
def a():
    values = np.random.randn(2,3)
    x0 = ['a','b']
    x1 = [11., 22., 33.]
    a = DimArray(values, axes=[x0, x1], dims=['d0','d1'])
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
