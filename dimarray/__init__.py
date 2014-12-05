""" Dimarray module
    ===============
"""
from __future__ import absolute_import

import warnings
from . import core
from . import config
from . import dataset
# from .lib import transform
from .lib import *

from .config import set_option, get_option, print_options, rcParams, rcParamsHelp
from .core import *
from .dataset import Dataset, stack_ds, concatenate_ds
from .datasets import get_datadir, get_ncfile
#from lib.transform import interp1d, interp2d, apply_recursive

# need to fix that
try:
    from dimarray.version import version as __version__
except:
    __version__ = "X.X.X"

try:
    from .io.nc import (read_nc, open_nc, summary_nc,
                        DatasetOnDisk)
    _ncio = True

except ImportError:
    _ncio = False
    msg = "Could not import netCDF4's package ==> I/O will not be available in this format"
    print msg
    #warnings.warn(ImportWarning(msg)) 
