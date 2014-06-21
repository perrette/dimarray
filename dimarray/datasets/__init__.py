from os.path import dirname, abspath, join

def get_datadir():
    """ Return directory name for the datasets
    """
    return dirname(abspath(__file__))

def get_ncfile(fname='cmip5.CSIRO-Mk3-6-0.nc'):
    """ Return one netCDF file
    """
    return join(get_datadir(),fname)
