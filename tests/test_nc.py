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

def test_read():
    """ test whether the reading of existing data works as expected
    """
    ncfile = get_ncfile('cmip5.CSIRO-Mk3-6-0.nc')
    ds = da.read_nc(ncfile)
    assert ds.keys() == [u'tsl', u'temp']
    assert ds.dims == ('time', 'scenario')
    assert ds.scenario.tolist() == ['historical', 'rcp26', 'rcp45', 'rcp60', 'rcp85']
    assert_equal( ds.time , np.arange(1850, 2301) )
    assert ds.time.dtype is np.dtype('int32') 
    assert ds.model == 'CSIRO-Mk3-6-0' # metadata
    assert isinstance(ds['tsl'], da.DimArray)
    assert ds['tsl'].ndim == 2
    assert ds['tsl'].shape == (451, 5)
    assert ds['tsl'].size == 451*5
    assert_almost_equal( ds['tsl'].values[0, :2] , np.array([ 0.00129744,  0.00129744]) )

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
