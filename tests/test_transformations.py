from __future__ import print_function
import sys
from warnings import warn
import unittest

import numpy as np
from numpy.testing import assert_allclose as _assert_allclose, assert_equal
import pytest

import dimarray as da
from dimarray.testing import assert_equal_dimarrays, assert_equal_datasets
from dimarray.tools import anynan

def assert_allclose(x, y, *args, **kwargs):
    _assert_allclose(x, y)
    if isinstance(x, da.DimArray) and isinstance(y, da.DimArray):
        for ax1, ax2 in zip(x.axes, y.axes):
            #assert assert_equal(ax1, ax2)
            assert ax1 == ax2


# return an array without nans
@pytest.fixture(scope='module', params=[float])
def vnonan(request):
    values = np.arange(4*3).reshape(4,3)
    values = np.asarray(values, dtype=request.param)
    time = 'time', np.arange(1950,1954) 
    lat = 'lat', np.linspace(-90,90,3)
    return da.DimArray(values, axes=[np.arange(1950,1954), np.linspace(-90,90,3)], dims=['time','lat'])

# return an array with nans
@pytest.fixture(scope='module')
def vnan(vnonan):
    vnan = vnonan.copy()
    vnan.values = np.asarray(vnan.values, dtype=float)
    vnan.values[[1,3],2] = np.nan
    return vnan

# return an array with and without nans
@pytest.fixture(scope='module', params=['nan','nonan'])
def v(request, vnan, vnonan):
    if request.param == 'nan':
        return vnan
    else:
        return vnonan

#@pytest.fixture()
#def axis():
#    return None

transforms = 'sum','prod','mean','var','std','min','max','ptp','all','any'

def pytest_generate_tests(metafunc):
    # e.g. test transformations with 3 axis values
    if 'axis' in metafunc.fixturenames:
        #metafunc.parametrize("axis", [None])
        metafunc.parametrize("axis", [None, 0, 1])

    if 'skipna' in metafunc.fixturenames:
        metafunc.parametrize("skipna", [True, False])
        #metafunc.parametrize("skipna", [True])

    if 'transform' in metafunc.fixturenames:
        #metafunc.parametrize("transform", ['sum'])
        metafunc.parametrize("transform", transforms)

    if 'use_bottleneck' in metafunc.fixturenames:
        #metafunc.parametrize("use_bottleneck", [False])
        metafunc.parametrize("use_bottleneck", [True, False])

def test_nonan(vnonan, axis, transform):
    " test transformations when no nan is present "
    v = vnonan
    axis_id, axis_nm = v._get_axis_info(axis)
    f1 = getattr(v, transform)
    f2 = getattr(v.values, transform)
    assert_allclose(f1(axis=axis_nm), f2(axis=axis_id))

def test_nan(vnan, axis, skipna, transform, use_bottleneck):
    "test transformations while nan are present: compare versus pandas"
    # use bottleneck?
    import dimarray.core.transform as tr
    orig_hasbottleneck = tr._hasbottleneck
    tr._hasbottleneck &= use_bottleneck

    try:
        axis_id, axis_nm = vnan._get_axis_info(axis)
        f1 = getattr(vnan, transform)
        if skipna:
            f2 = getattr(vnan.to_MaskedArray(), transform)
        else:
            f2 = getattr(vnan.values, transform)

        res1 = f1(skipna=skipna, axis=axis)
        res2 = f2(axis=axis)
        if isinstance(res2, np.ma.MaskedArray):
            res2 = res2.filled(np.nan)
        #if not np.isscalar(res1): res1 = res1.values
        #if not np.isscalar(res2): res2 = res2.values
        assert_allclose( res1, res2)

        # also try versus pandas
        if axis is None: return # can't flatten an array
        try:
            import pandas as pd
        except ImportError:
            print("pandas is not installed, can't test transforms with nan")
            return

        f2 = getattr(vnan, transform)
        res2 = f2(axis=axis, skipna=skipna)
        assert_allclose( res1, res2)
            

    finally:
        tr._hasbottleneck = orig_hasbottleneck # set back to normal value

#
# now test a few special functions
#
def argsearch(vnonan, axis):
    # argmin, argmax: locate the minimum and maximum of an array
    if axis is None:
        assert_allclose(v[v.argmin()] , v.min())
        assert_allclose(v[v.argmax()] , v.max())
    else:
        v.argmin(axis=axis_nm)
        v.argmax(axis=axis_nm)

def test_median(v, axis, use_bottleneck):

    if not anynan(v):
        assert_allclose(v.median(axis=axis), np.median(v.values, axis=axis))
    else:
        assert_allclose(v.median(axis=axis, skipna=True), np.ma.median(v.to_MaskedArray(), axis=axis))

def test_diff(vnonan):
    v = vnonan
    v.diff(axis='time', keepaxis=False)
    v.diff(axis=0, keepaxis=False, scheme='centered')
    v.diff(axis=0, keepaxis=False, scheme='backward')
    v.diff(axis=0, keepaxis=False, scheme='forward')
    v.diff(axis=0, keepaxis=True, scheme='backward')
    v.diff(axis=0, keepaxis=True, scheme='forward')
    v.diff(n=2,axis=('time'), scheme='centered')

try:
    import pandas
    PANDAS_INSTALLED = True
except:
    PANDAS_INSTALLED = False

@pytest.mark.skipif(not PANDAS_INSTALLED,
                                        reason="requires pandas")
def test_vs_pandas_special(vnonan):

    #np.random.seed(0)
    #v = da.DimArray(np.random.randn(5,7), {'time':np.arange(1950,1955), 'lat':np.linspace(-90,90,7)})
    v = vnonan
    assert_allclose(v.std(ddof=1, axis=0).values, v.to_pandas().std().values), "std vs pandas"
    assert_allclose(v.var(ddof=1, axis=0).values, v.to_pandas().var().values), "var vs pandas"
    assert_allclose(v.cumsum(axis=0).values , v.to_pandas().cumsum().values), "pandas: cumsum failed"
    assert_allclose(v.cumprod(axis=0).values , v.to_pandas().cumprod().values), "pandas: cumprod failed"
    # assert_allclose(v.diff(axis=0, keepaxis=True).cumsum(axis=0, skipna=True).values, v.to_pandas().diff().cumsum(skipna=True).values), "diff-cumsum failed"

    # TEST diff
    res = v.diff(axis=0, keepaxis=True) 
    assert_allclose(res, v.to_pandas().diff()), 'diff failed'

def test_operations():
    a = da.DimArray([[1,2,3],[3,4,5]],dims=('x0','x1'))
    assert_allclose(a , a)
    assert_allclose(a+2 , a + np.ones(a.shape)*2)
    assert_allclose(a+a , a*2)
    assert_allclose(a*a , a**2)
    assert_allclose((a - a.values) , a - a)


class TestInterp(unittest.TestCase):

    def setUp(self):
        size = 4
        index = np.array([1,4,3,2.])
        dima = da.DimArray(index, axes=[index])
        self.index = index
        self.dima = dima
        self.size = size

        newindex = self.index.copy()
        np.random.shuffle(newindex)
        self.newindex = newindex

    def testVsNumpy(self):
        dima = self.dima
        newindex = self.newindex

        dima_sorted = dima.sort_axis()

        expected = np.interp(newindex, dima_sorted.x0, dima_sorted.values)  # np.interp, pure numpy

        actual = dima_sorted.interp_axis(newindex, issorted=True)
        _assert_allclose(actual.values, expected) 
        _assert_allclose(actual.axes[0], newindex) 

        actual = dima.interp_axis(newindex) # additional-presorting
        _assert_allclose(actual.values, expected) 
        _assert_allclose(actual.axes[0], newindex) 

    def testWeights(self):
        " a bit more than just numpy.interp wrapper, re-used weights..."
        dima = self.dima
        newindex = self.newindex

        dima2d = dima.newaxis('bla', pos=1)

        dima_s = dima.sort_axis()
        expected = np.interp(newindex, dima_s.x0, dima_s.values)  # np.interp, pure numpy
        actual = dima2d.interp_axis(newindex).squeeze().values  # self-computed weights using np.interp

        _assert_allclose(actual, expected)

    def test3d(self):
        " test 3-D interp"
        dima_3d = self.dima.newaxis('bla',np.arange(2)).newaxis('bli',np.arange(3)) # should be 10^4 slower
        interpolated = dima_3d.interp_axis(self.newindex, axis=-1)
        for i in range(2):
            actual = interpolated.ix[i,0]
            expected = dima_3d.ix[i,0].interp_axis(self.newindex) # use np.interp directly
            assert_equal_dimarrays(actual, expected)

    def testDataset(self):
        " make sure the dataset version is consistent "
        c = self.dima.sort_axis()
        print(c.dims)
        c.dims = ('x0_sorted',)
        ds = da.Dataset(a=self.dima, b=self.dima*2, c=c)

        actual = ds.interp_axis(self.newindex, axis='x0')
        expected = da.Dataset(**{k:ds[k].interp_axis(self.newindex, axis='x0') if 'x0' in ds[k].dims else ds[k] for k in ds.keys()})
        
        print('for a')
        print(self.newindex)
        print(actual['a'])
        print(expected['a'])
        print('for b')
        print(actual['b'])
        print(expected['b'])
        print('for c')
        print(c)
        print(actual['c'])
        print(expected['c'])

        print("")
        print("datasets assert")

        assert_equal_datasets(actual, expected)
