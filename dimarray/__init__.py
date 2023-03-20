""" Dimarray module
    ===============
"""

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


try:
    from .io.nc import (read_nc, open_nc, summary_nc,
                        DatasetOnDisk)
    _ncio = True

except ImportError:
    _ncio = False
    msg = "Could not import netCDF4's package ==> I/O will not be available in this format"
    print(msg)
    #warnings.warn(ImportWarning(msg)) 


from pathlib import Path
__version__ = open(Path(__file__).parent / "_version.py").read()
