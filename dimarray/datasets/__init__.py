import zipfile,os.path
import warnings
from os.path import dirname, abspath, join, exists

# current directory
DIREC=dirname(abspath(__file__))

# unzip data the first time this module is imported
def get_datadir():
    """ Return directory name for the datasets
    """
    return join(DIREC, 'data')

def get_ncfile(fname='cmip5.CSIRO-Mk3-6-0.nc'):
    """ Return one netCDF file
    """
    return join(get_datadir(),fname)
