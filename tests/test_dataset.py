"""
"""
import numpy as np
from numpy.testing import assert_array_equal
import unittest
import pytest
import dimarray as da
from dimarray import Dataset


class Structure(unittest.TestCase):
    """ Test dataset's structure
    """
    def setUp(self):
        """ initialize a dataset
        """
        a = da.zeros(axes=[['a','b'],[11.,22.,33.]], dims=['d0','d1'])
        b = da.ones(axes=[['a','b']], dims=['d0'])
        c = da.zeros(axes=[[1,2,3]], dims=['d2'])

        self.ds = Dataset([('aa',a),('cc',c)]) # like an ordered dict
        self.ds['bb'] = b  # axis 'd0' will be checked
        self.ds['cc'] = c  # axis 'd0' will be checked

    def test_repr(self):

        # test repr
        expected_repr = """ 
Dataset of 3 variables
0 / d0 (2): a to b
1 / d1 (3): 11.0 to 33.0
2 / d2 (3): 1 to 3
aa: ('d0', 'd1')
cc: ('d2',)
bb: ('d0',)
    """.strip()

        assert repr(self.ds) == expected_repr

    def test_axes(self):

        ds = self.ds

        # "shallow" equality
        assert set(ds.dims) == set(('d0','d1','d2'))
        assert np.all( ds.d0 == ['a','b'])
        assert np.all( ds['bb'].d0 == ['a','b'])

        # test copy/ref behaviour
        assert ds.axes['d0'] is ds['aa'].axes['d0']
        assert ds.axes['d0'] is ds['bb'].axes['d0']
        assert ds.axes['d2'] is ds['cc'].axes['d2']

        # delete
        assert self.ds.dims == ('d0','d1','d2')
        del self.ds['cc'] 
        assert self.ds.dims == ('d0','d1'), "axis not deleted"

        # modify datasets' axis values
        ds.axes['d0'][1] = 'yo'
        assert ds['aa'].d0[1] == 'yo'
        assert ds['bb'].d0[1] == 'yo'

        # modify axis names, single
        ds.axes['d0'].name = 'new_d0'
        assert ds.dims == ('new_d0', 'd1')
        assert ds['aa'].dims == ('new_d0', 'd1')
        assert ds['bb'].dims == ('new_d0',)

        # modify axes names (dims), bulk
        ds.dims = ('d0n','d1n')
        assert ds.dims == ('d0n','d1n')
        assert ds['aa'].dims == ('d0n','d1n')
        assert ds['bb'].dims == ('d0n',)

    def test_copy(self):

        ds = self.ds
        ds.units = 'myunits'

        ds2 = ds.copy()
        assert isinstance(ds2, Dataset)
        assert ds2.axes == ds.axes
        assert ds2.axes is not ds.axes
        assert hasattr(ds2, 'units')
        assert ds2.units == ds.units

        assert ds == ds2

        ds2['aa']['b',22.] = -99
        assert np.all(ds['aa'] == ds2['aa']) # shallow copy ==> elements are just references

        ds2['bb'] = ds['bb'] + 33 
        assert np.all(ds['bb'] != ds2['bb']), 'shallow copy'

    def test_reindexing(self):

        ds = self.ds
        d = da.ones(axes=[['a','b','c'],[33,44]], dims=['d0','d5'])

        # raises exception when setting an array with unaligned axes
        with pytest.raises(ValueError):
            ds['dd'] = d

        # that works
        ds['dd'] = d.reindex_like(ds)
        assert np.all( ds.d0 == ['a','b'])

        # on-the-fly reindexing at initialization
        ds2 = Dataset( aa=ds['aa'], dd=d)

        assert np.all( ds2.d0 == ['a','b','c'])
        assert np.all( ds2['aa'].d0 == ['a','b','c'])


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


#
# Older tests
#
#@pytest.mark.last
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
