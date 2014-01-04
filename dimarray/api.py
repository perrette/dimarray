""" User-interface, should be relatively stable
"""
from config import set_option, get_option, Config
from da import DimArray, array, Axis, Axes
from dataset import Dataset
from lib.align import broadcast_arrays, align_dims, align_axes
from lib.transform import interp1d, interp2d, apply_recursive

try:
    from io.nc import read as read_nc, summary as summary_nc
    _ncio = True
except:
    raise
    _ncio = False
    print "ncio not imported"
    pass
