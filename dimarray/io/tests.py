""" Test module for nc
"""
import os
from warnings import warn

import numpy as np

import dimarray  as da
from dimarray.testing import testmod
from dimarray import DimArray, summary_nc, read_nc

curdir = os.path.dirname(__file__)
testdata = os.path.join(curdir, 'testdata')

def test_ncio():
    a = DimArray([1,2], dims='xx0')
    b = DimArray([3,4,5], dims='xx1')
    fname = os.path.join(testdata, "test.nc")
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


#def _main(**kwargs):
#    """ go to testdata and make sure the "test.nc" file exist
#    """
#    curdir = os.path.abspath(os.getcwd()) # current directory
#    os.chdir(testdata) # change to testdata
#    res = testmod(nc, **kwargs)
#    os.chdir(curdir) # come back to current directory
#    return res

def main(**kw):
    try:
        test_ncio()
    except RuntimeError, msg:
        warn("NetCDF test failed: {}".format(msg))


if __name__ == "__main__":
    main()
