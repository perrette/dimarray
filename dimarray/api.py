""" User-interface, should be relatively stable
"""
from core import Dimarray, array
from axes import Axis, Axes
from dataset import Dataset
from lib.reindex import align_axes, interp2d
from lib.reshape import broadcast_arrays, align_dims

try:
    from io.nc import read as read_nc, summary as summary_nc
    _ncio = True
except:
    raise
    _ncio = False
    print "ncio not imported"
    pass
