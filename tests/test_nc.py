""" Test module for nc
"""
import os
from warnings import warn
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

import dimarray  as da
from dimarray import DimArray, summary_nc, read_nc, open_nc, write_nc, get_ncfile

curdir = os.path.dirname(__file__)

# parameterize test: 
# - test writing netcdf files for arrays of various types
# - 
def pytest_generate_tests(metafunc):
    if 'ncvar_shape' in metafunc.fixturenames:
        metafunc.parametrize("ncvar_shape", [(2,3), ])
    if 'ncvar_type' in metafunc.fixturenames:
        metafunc.parametrize("ncvar_type", [int, float, bool])
    if 'ncdim_type' in metafunc.fixturenames:
        metafunc.parametrize("ncdim_type", [int, float, str, unicode, object])
    # if 'ncattr_type' in metafunc.fixturenames:
    #     metafunc.parametrize("ncattr_type", [str, unicode, int, float, list, np.ndarray], indirect=True)
    if 'seed' in metafunc.fixturenames:
        metafunc.parametrize("seed", [1], indirect=True)

def generate_array(shape, dtype):
    """ create an array of desired type """
    values = np.random.rand(*shape)
    if dtype is bool:
        values = values > 0.5
    else:
        values = np.asarray(values, dtype=dtype)
    return values

def generate_array_axis(size, dtype, regular=True):
    """ create an axis of desired type """
    values = np.arange(size) + 10*np.random.rand() # random offset
    if not regular:
        values = np.random.shuffle(values)
    return np.asarray(values, dtype=dtype)

def generate_metadata(types=[str, unicode, int, float, list, np.ndarray]):
    meta = {}
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for i, t in enumerate(types):
        if t in (str, unicode): 
            val = t('some string')
        elif t in (list, np.ndarray):
            val = t([1,2])
        else:
            val = t() # will instantiate some default value

        meta[letters[i]] = val
    return meta

@pytest.fixture()
def dim_array(ncvar_shape, ncvar_type, ncdim_type, seed=None):
    """ Parameterized DimArray
    """
    np.random.seed(seed) # set random generator
     
    # create an array of desired type
    values = generate_array(ncvar_shape, dtype=ncvar_type)

    # create axes
    axes = [generate_array_axis(s, ncdim_type) for s in ncvar_shape]

    # create metadata
    meta = generate_metadata()

    return da.DimArray(values, axes, **meta)

class ReadTest(object):
    """ Ancestor class for reading netCDF files, to be inherited, is not executed as test
    """
    ncfile = get_ncfile('cmip5.CSIRO-Mk3-6-0.nc')

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_csiro(self):
        " manual check for the CSIRO dataset "
        ds = self.ds 
        assert ds.keys() == [u'tsl', u'temp']
        assert ds.dims == ('time', 'scenario')
        assert ds.scenario.tolist() == ['historical', 'rcp26', 'rcp45', 'rcp60', 'rcp85']
        assert_equal( ds.time , np.arange(1850, 2301) )
        assert ds.time.dtype is np.dtype('int32') 
        assert ds.model == 'CSIRO-Mk3-6-0' # metadata
        assert ds['tsl'].ndim == 2
        assert ds['tsl'].shape == (451, 5)
        assert ds['tsl'].size == 451*5
        assert_almost_equal( ds['tsl'].values[0, :2] , np.array([ 0.00129744,  0.00129744]) )

class TestReadReadNC(ReadTest):
    @classmethod
    def setup_class(cls):
        cls.ds = da.read_nc(cls.ncfile)

class TestReadOpenNC(ReadTest):
    " read netCDF file with open_nc "

    @classmethod
    def setup_class(cls):
        cls.ds = da.open_nc(cls.ncfile)

    @classmethod
    def teardown_class(cls):
        cls.ds.close()

    def test_access(self):
        ds = self.ds
        assert isinstance(ds['tsl'][:], da.DimArray)
        assert isinstance(ds['tsl'].values[:], np.ndarray)
        expected = ds.ds.variables['tsl'][:] # netCDF4
        assert_equal( ds['tsl'][:].values , expected )
        assert_equal( ds['tsl'].values[:] , expected ) # alternative (more efficient) form

    def test_indexing(self):

        size0 = len(self.ds.ds.dimensions.values()[0])

        boolidx = np.random.rand(size0) > 0.5
        indices = [5, -1, [0, 1, 2], boolidx, (0,1), slice(None), slice(2,10,2)]

        ds = self.ds

        # position indexing
        for idx in indices:
            print "Test index: ", idx
            expected = ds.ds.variables['tsl'][idx] # netCDF4
            actual = ds['tsl'].ix[idx].values
            assert_equal( expected, actual)

        # label indexing
        labels = [(idx, self.ds.ds.variables['time'][idx]) for idx in indices if type(idx) not in (tuple, slice, np.ndarray)]
        labels.append(((0,1), (self.ds.ds.variables['time'][0], self.ds.ds.variables['scenario'][1])))
        labels.append((slice(None), slice(None)))
        labels.append((slice(2,10,2), slice(self.ds.ds.variables['time'][2],self.ds.ds.variables['time'][10-1],2)))
        labels.append((boolidx, boolidx))

        for idx, lidx in labels:
            print "Test labelled index:",idx,':',lidx
            expected = ds.ds.variables['tsl'][idx] # netCDF4
            actual = ds['tsl'][lidx].values
            assert_almost_equal( expected, actual)

def test_io(dim_array, tmpdir): 

    fname = tmpdir.join("test.nc").strpath # have test.nc in some temporary directory

    a = DimArray([1,2], dims='xx0')
    b = DimArray([3,4,5], dims='xx1')
    a.write_nc(fname,"a", mode='w')
    b.write_nc(fname,"b", mode='a')
    try:
        b.write_nc(fname.replace('.nc','netcdf3.nc'),"b", mode='w', format='NETCDF3_CLASSIC')
    except Exception, msg:
        warn("writing as NETCDF3_CLASSIC failed (known bug on 64bits systems): {msg}".format(msg=repr(msg)))

    data = read_nc(fname)
    assert(np.all(data['a'] == a))
    assert(np.all(data['b'] == b))
    ds = da.Dataset(a=a, b=b)
    for k in ds:
        assert(np.all(ds[k] == data[k]))

def test_open_nc(tmpdir):
    pass

def main(**kw):
    try:
        test_ncio()
    except RuntimeError, msg:
        warn("NetCDF test failed: {}".format(msg))

if __name__ == "__main__":
    main()
