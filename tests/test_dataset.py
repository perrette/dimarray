"""
"""
import numpy as np
from numpy.testing import assert_array_equal
import unittest
import pytest
import dimarray as da
from dimarray import Dataset
from dimarray.testing import assert_equal_axes, assert_equal_datasets


class TestStructure(unittest.TestCase):
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
0 / d0 (2): 'a' to 'b'
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

# was after __setitem__ in Dataset ==> prob enough with the above
# # now just checking 
# test_internal = super(Dataset, self).__getitem__(key)
# for ax in test_internal.axes:
#     assert self.axes[ax.name] is ax


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



@pytest.fixture
def ax0():
    return da.Axis([11.,22.,33.], name="dim0")
@pytest.fixture
def ax1():
    return da.Axis(["a","b"], name="dim1")
@pytest.fixture
def v0(ax0):
    return da.DimArray([111., 222., 333.], axes=[ax0])
@pytest.fixture
def v1(ax1):
    return da.DimArray([4, 5], axes=[ax1])
@pytest.fixture
def v2(ax0, ax1):
    return da.DimArray([[1,2],[3,4],[5, 6]], axes=[ax0, ax1])
@pytest.fixture
def ds(v0, v1, v2):
    return da.Dataset(v0=v0, v1=v1, v2=v2)

# Test changing axes properties
def test_update_axis(ds, ax0):
    assert isinstance(ds.axes, da.dataset.DatasetAxes)
    dim0_2 = da.Axis(ax0.values*2, "dim0")
    ds.axes["dim0"] = dim0_2
    assert_equal_axes(ds.axes["dim0"], dim0_2)
    assert_equal_axes(ds["v0"].axes["dim0"], dim0_2)
    assert_equal_axes(ds["v2"].axes["dim0"], dim0_2)


# Test initialization of a DimArray with Dataset axes...
def test_dataset_derived_axes(ds):
    shp = list(ax.size for ax in ds.axes)
    dima = da.DimArray(np.ones(shp), axes=ds.axes)
    assert_equal_axes(ds.axes[0], dima.axes[0])
    assert_equal_axes(ds.axes[1], dima.axes[1])

# Append axis in an otherwise empty dataset
def test_append_axis():
    ds = da.Dataset()
    ds.axes.append(da.Axis([10,20,30], 'myaxis'))
    assert list(ds.keys()) == []
    assert ds.dims == ("myaxis",)

# Test indexing dataset
def test_indexing():
    x = np.array([0, 0.1, 0.2, 0.3])
    y0 = 2*x
    y1 = x**2

    ds = da.Dataset({
        'y0':da.DimArray(y0, axes=[x], dims=['dim0']),
        'y1':da.DimArray(y1, axes=[x], dims=['dim0']),
    })

    test_indices = [slice(1,3), [1,3]]
    for ix in test_indices:

        ds_expected = da.Dataset({
            'y0':da.DimArray(y0[ix], axes=[x[ix]], dims=['dim0']),
            'y1':da.DimArray(y1[ix], axes=[x[ix]], dims=['dim0']),
        })

        # test position indexing
        assert_equal_datasets(ds.ix[ix], ds_expected)

        # pandas-compatible iloc
        assert_equal_datasets(ds.iloc[ix], ds_expected)

        # test label indexing with pandas-compatible loc
        lab_ix = x[ix]
        assert_equal_datasets(ds.loc[lab_ix], ds_expected)
