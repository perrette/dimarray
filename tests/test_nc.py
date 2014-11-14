""" Test module for nc
"""
import os
from warnings import warn
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

import dimarray  as da
from dimarray import DimArray, summary_nc, read_nc, open_nc, write_nc, get_ncfile
from dimarray.testing import (assert_equal_dimarrays, assert_equal_datasets, assert_equal_axis,
                              create_dataset, create_dimarray)

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
        metafunc.parametrize("seed", [1])

# to use the function above in a fixture
@pytest.fixture()
def dim_array(ncvar_shape, ncvar_type, ncdim_type, seed):
    return create_dimarray(ncvar_shape, dtype=ncvar_type, axis_dtypes=ncdim_type, seed=seed)

@pytest.fixture()
def data_set(seed):
    return create_dataset(seed=seed)


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
        assert ds.scenario[:].tolist() == ['historical', 'rcp26', 'rcp45', 'rcp60', 'rcp85']
        assert_equal( ds.time , np.arange(1850, 2301) )
        assert ds.time.dtype is np.dtype('int32') 
        print ds.attrs
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
        expected = ds.nc.variables['tsl'][:] # netCDF4
        assert_equal( ds['tsl'][:].values , expected )
        assert_equal( ds['tsl'].values[:] , expected ) # alternative (more efficient) form

    def test_indexing(self):

        size0 = len(self.ds.nc.dimensions.values()[0])
        boolidx = np.random.rand(size0) > 0.5
        indices = [5, -1, [0, 1, 2], boolidx, (0,1), slice(None), slice(2,10,2)]

        ds = self.ds

        # position indexing
        for idx in indices:
            # print "Test index: ", idx
            expected = ds.nc.variables['tsl'][idx] # netCDF4
            actual = ds['tsl'].ix[idx]
            actual = getattr(actual, "values", actual)
            assert_equal(actual, expected)

        # label indexing
        labels = [(idx, self.ds.nc.variables['time'][idx]) for idx in indices if type(idx) not in (tuple, slice, np.ndarray)]
        labels.append(((0,1), (self.ds.nc.variables['time'][0], self.ds.nc.variables['scenario'][1])))
        labels.append((slice(None), slice(None)))
        labels.append((slice(2,10,2), slice(self.ds.nc.variables['time'][2],self.ds.nc.variables['time'][10-1],2)))
        labels.append((boolidx, boolidx))

        for idx, lidx in labels:
            print "Test labelled index:",idx,':',lidx
            expected = ds.nc.variables['tsl'][idx] # netCDF4
            actual = ds['tsl'][lidx]
            actual = getattr(actual, "values", actual)
            assert_equal(actual, expected)

class TestIO(object):

    @classmethod
    def setup_class(cls):
        cls.ds_var = create_dataset()

        cls.tmpdir = '/tmp'
        # cls.ncfile = cls.tmpdir.join("test.nc").strpath # have test.nc in some temporary directory
        cls.ncfile = os.path.join(cls.tmpdir, "test.nc")
        print "Write test netCDF file to",cls.ncfile
        cls.ds_var.write_nc(cls.ncfile, mode='w')
        os.system("ncdump -v str_0d "+cls.ncfile)

    @classmethod
    def teardown_class(cls):
        pass

    # @pytest.mark.tryfirst
    # def test_write_nc_whole(self, tmpdir):
    #     # write whole dataset to disk
    #     self.tmpdir = tmpdir
    #     self.ncfile = self.tmpdir.join("test.nc").strpath # have test.nc in some temporary directory
    #     print "Write test netCDF file to",self.ncfile
    #     self.ds_var.write_nc(self.ncfile, mode='w')

    def test_read_nc_whole(self):
        " check that the netCDF file matches what has been written "
        os.system("ncdump -v str_0d "+self.ncfile)
        actual = da.read_nc(self.ncfile)
        os.system("ncdump -v str_0d "+self.ncfile)
        assert_equal_datasets(actual, self.ds_var)

    def test_basic_write_and_read(self, tmpdir):
        " try writing variable-by-variable with open_nc "
        # with da.open_nc(self.ncfile) as ds_disk:
        ds_var = self.ds_var
        self.ncfile2 = tmpdir.join("test2.nc").strpath # have test.nc in some temporary directory
        with da.open_nc(self.ncfile2, mode='w', clobber=True) as ds_disk:
            for k in ds_var.keys():
                ds_disk[k] = ds_var[k]
                assert_equal_dimarrays(ds_disk[k].read(), ds_var[k]) # immediate test, before closing file
            # add dataset metadata
            ds_disk._metadata(ds_var._metadata())

        # read the whole dataset and check for equality
        with da.open_nc(self.ncfile2) as ds_disk:
            ds = ds_disk.read()
        actual, expected = ds, ds_var
        assert_equal_datasets(actual, expected)

    def test_read_per_variable(self):
        # read each variable individually
        with da.open_nc(self.ncfile) as ds_disk:
            for k in ds_disk.keys():
                print 'read', k, self.ncfile
                assert_equal_dimarrays(ds_disk[k][:], self.ds_var[k])
                assert_equal_dimarrays(ds_disk[k].read(), self.ds_var[k])

    def test_read_position_index(self):
        # try opening bits of the first variable
        boolidx = np.random.rand(self.ds_var.axes[0].size) > 0.5
        indices = [1, -1, [0, 1], boolidx, (0,1), slice(None), slice(1,None,2)]

        # position indexing
        with da.open_nc(self.ncfile) as ds_disk:
            v0 = ds_disk.keys()[0]
            for idx in indices:
                expected = self.ds_var[v0].ix[idx]
                actual = ds_disk[v0].ix[idx]
                assert_equal_dimarrays(actual, expected)

    def test_read_label_index(self):
        # label indexing: test against DimArray indexing
        boolidx = np.random.rand(self.ds_var.axes[0].size) > 0.5
        indices = [1, -1, [0, 1], boolidx, (0,1), slice(None), slice(1,None,2)]
        labels = [self.ds_var.axes[0].values[idx] for idx in indices if type(idx) not in (tuple, slice, np.ndarray)]
        labels.append((self.ds_var.axes[0].values[0], self.ds_var.axes[1].values[1]))
        labels.append((slice(None), slice(None)))
        labels.append(slice(self.ds_var.axes[0].values[1],None,2))
        labels.append(boolidx)

        with da.open_nc(self.ncfile) as ds_disk:
            v0 = ds_disk.keys()[0]
            for lidx in labels:
                expected = self.ds_var[v0][lidx]
                actual = ds_disk[v0][lidx]
                assert_equal_dimarrays(actual, expected)

    # def test_write(self):
    #     cls.ds_disk = open_nc(cls.ncfile, mode='w', clobber=True) # open netCDF file for writing
    #     cls.ds.close()

def test_io(dim_array, tmpdir): 

    fname = tmpdir.join("test.nc").strpath # have test.nc in some temporary directory

    a = DimArray([1,2], dims=['xx0'])
    b = DimArray([3,4,5], dims=['xx1'])
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
