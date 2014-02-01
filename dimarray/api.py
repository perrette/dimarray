""" User-interface, should be relatively stable
"""
import core
import config
import dataset
import lib.transform

from config import set_option, get_option, Config
from core import *
from dataset import Dataset
from lib.transform import interp1d, interp2d, apply_recursive

try:
    from io.nc import read as read_nc, summary as summary_nc, NCDataset as ncio
    _ncio = True
except:
    raise
    _ncio = False
    print "ncio not imported"
    pass
