""" Dimarray module
    ===============
"""
import warnings
import core
import config
import dataset
import lib.transform

from config import set_option, get_option, print_options, rcParams, rcParamsHelp
from core import *
from dataset import Dataset, stack_ds, concatenate_ds
from datasets import get_datadir, get_ncfile
#from lib.transform import interp1d, interp2d, apply_recursive
from lib import *

from dimarray.version import version as __version__

try:
    from io.nc import read_nc, summary_nc, write_nc #, read_nc_axes
    _ncio = True

except ImportError:
    _ncio = False
    msg = "Could not import netCDF4's package ==> I/O will not be available in this format"
    print msg
    #warnings.warn(ImportWarning(msg)) 
