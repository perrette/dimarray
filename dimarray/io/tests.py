""" Test module for nc
"""
import os

import dimarray 
from dimarray.testing import testmod
import nc

curdir = os.path.dirname(__file__)
testdata = os.path.join(curdir, 'testdata')

def _main(**kwargs):
    """ go to testdata and make sure the "test.nc" file exist
    """
    curdir = os.path.abspath(os.getcwd()) # current directory
    os.chdir(testdata) # change to testdata
    res = testmod(nc, **kwargs)
    os.chdir(curdir) # come back to current directory
    return res

def main(**kw):
    pass

if __name__ == "__main__":
    main()
