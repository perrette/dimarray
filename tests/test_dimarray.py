""" Test basic dimarray behaviour (copy, attributes)
"""
from __future__ import print_function
import pytest
import numpy as np
from dimarray import DimArray, Axis
from dimarray.testing import create_array, create_dimarray

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

def test_cast():
    # a = asarray(a, dtype=t0)
    # b = asarray(b, dtype=t1)
    # a.put(b, cast=True)
    # a.dtype.kind == t2
    # test list of tuples (t0, t1, t2)
    tests = [
        (int, float, 'f'),
        ('i', 'f', 'f'),
        ('int32', 'float32', 'f'),
        ('int64', 'float32', 'f'),
        ('int64', 'float64', 'f'),
        ('int32', 'float64', 'f'),
        ('int32', 'float64', 'f'),
        (float, object, 'O'),
        ('f', 'O', 'O'),
        ('f', str, 'O'),
        ('i', str, 'O'),
        ('S', 'U', 'U'),
        ('U', 'S', 'U'),
    ]
    for test in tests:
        print(test)
        a = create_dimarray((5,), dtype=test[0])
        b = create_dimarray((), dtype=test[1])
        # a.ix[0] = b[()]
        # print b
        print(a.dtype.kind, b.dtype.kind)
        a.put((),b, cast=True, indexing='position')
        # print a
        assert a.dtype.kind == test[2]


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
