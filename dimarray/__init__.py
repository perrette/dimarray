""" 
"""
import warnings
import core
import config
import dataset
import lib.transform

from config import set_option, get_option, print_options, rcParams, rcParamsHelp
from core import *
from dataset import Dataset, stack_ds, concatenate_ds
#from lib.transform import interp1d, interp2d, apply_recursive
from lib import *

try:
    from io.nc import read as read_nc, summary as summary_nc
    _ncio = True

except ImportError:
    _ncio = False
    msg = "Could not import netCDF4's package ==> I/O will not be available in this format"
    print msg
    #warnings.warn(ImportWarning(msg)) 
