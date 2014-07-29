"""
"""
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import dimarray as da
from dimarray import Dataset


def test_init():

    # test initialization
    a = da.zeros(axes=[['a','b'],[11.,22.,33.]], dims=['d0','d1'])
    b = da.ones(axes=[['a','b']], dims=['d0'])
    c = da.zeros(axes=[[1,2,3]], dims=['d2'])
    d = da.ones(axes=[['a','b','c']], dims=['d0'])

    ds = Dataset(a=a, b=b, c=c, d=d)
    assert set(ds.dims) == set(('d0','d1','d2'))
    assert_array_equal( ds.d0, ['a','b','c'])

    # test repr
    expected_repr = """ 
Dataset of 4 variables
0 / d0 (3): a to c
1 / d1 (3): 11.0 to 33.0
2 / d2 (3): 1 to 3
a: ('d0', 'd1')
c: ('d2',)
b: ('d0',)
d: ('d0',)
""".strip()

    assert repr(ds) == expected_repr

def test_getsetdel():
    a = da.zeros(axes=[['a','b'],[11.,22.,33.]], dims=['d0','d1'])
    b = da.ones(axes=[['a','b']], dims=['d0'])
    c = da.zeros(axes=[[1,2,3]], dims=['d2'])
    d = da.ones(axes=[['a','b','c']], dims=['d0'])

    # first assingment
    ds = Dataset()
    ds['a'] = a
    assert ds.axes == a.axes, "this should be equal"
    assert ds.axes is not a.axes, 'this should be a copy'

    ds['b'] = b
    ds['c'] = c
    del ds['a'] 
    assert ds.dims == ('d0','d2'), "d1 should have been deleted, got {}".format(ds.dims)

    # this should raise value error because axes are not aligned
    with pytest.raises(ValueError):
        ds['d'] = d 

    ds['d'] = d.reindex_like(ds)

def test_axes():
    """ test copy / ref behaviour of axes
    """
    a = da.zeros(axes=[['a','b'],[11.,22.,33.]], dims=['d0','d1'])
    b = da.ones(axes=[['a','b']], dims=['d0'])

    ds = Dataset(a=a, b=b)
    ds.axes['d0'][1] = 'yo'
    assert ds['a'].d0[1] == 'yo'
    assert ds['b'].d0[1] == 'yo'

    # individual axes are equal but copies
    assert ds['a'].axes['d0'] == ds.axes['d0']
    assert ds['a'].axes['d0'] is not ds.axes['d0']
    assert ds['b'].axes['d0'] == ds.axes['d0']
    assert ds['b'].axes['d0'] is not ds.axes['d0']

def test_metadata():
    ds = Dataset()
    a = da.zeros(shape=(4,5))
    a.units = 'meters'
    ds['a'] = a

    # metadata are conserved
    assert ds['a'].units == 'meters'

    # copy / reference behaviour 
    ds['a'].units = 'millimeters'
    assert ds['a'].units == 'millimeters' 

def test_copy():

    a = da.zeros(axes=[['a','b'],[11.,22.,33.]], dims=['d0','d1'])
    b = da.ones(axes=[['a','b']], dims=['d0'])
    ds = Dataset(a=a, b=b)
    ds.units = 'myunits'

    ds2 = ds.copy()
    assert isinstance(ds2, Dataset)
    assert ds2.axes == ds.axes
    assert ds2.axes is not ds.axes
    assert hasattr(ds2, 'units')
    assert ds2.units == ds.units

    assert ds == ds2

    ds2['a']['b',22.] = -99
    assert np.all(ds['a'] == ds2['a']) # shallow copy ==> elements are just references

    ds2['b'] = ds['b'] + 33 
    assert np.all(ds['b'] != ds2['b']), 'shallow copy'


def test():
    """ Test Dataset functionality

    >>> data = test() 
    >>> data['test2'] = da.DimArray([0,3],('source',['greenland','antarctica'])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: axes values do not match, align data first.                            
    Dataset: source(1)=greenland:greenland, 
    Got: source(2)=greenland:antarctica
    >>> data['ts']
    dimarray: 5 non-null elements (5 null)
    0 / time (10): 1950 to 1959
    array([  0.,   1.,   2.,   3.,   4.,  nan,  nan,  nan,  nan,  nan])
    >>> data.to_array(axis='items')
    dimarray: 12250 non-null elements (1750 null)
    0 / items (4): mymap to test
    1 / lon (50): -180.0 to 180.0
    2 / lat (7): -90.0 to 90.0
    3 / time (10): 1950 to 1959
    4 / source (1): greenland to greenland
    array(...)
    """
    import dimarray as da
    axes = da.Axes.from_tuples(('time',[1, 2, 3]))
    ds = da.Dataset()
    a = da.DimArray([[0, 1],[2, 3]], dims=('time','items'))
    ds['yo'] = a.reindex_like(axes)

    np.random.seed(0)
    mymap = da.DimArray.from_kw(np.random.randn(50,7), lon=np.linspace(-180,180,50), lat=np.linspace(-90,90,7))
    ts = da.DimArray(np.arange(5), ('time',np.arange(1950,1955)))
    ts2 = da.DimArray(np.arange(10), ('time',np.arange(1950,1960)))

    # Define a Dataset made of several variables
    data = da.Dataset({'ts':ts, 'ts2':ts2, 'mymap':mymap})
    #data = da.Dataset([ts, ts2, mymap], keys=['ts','ts2','mymap'])

    assert np.all(data['ts'].time == data['ts2'].time),"Dataset: pb data alignment" 

    data['test'] = da.DimArray([0],('source',['greenland']))  # new axis
    #data

    return data
