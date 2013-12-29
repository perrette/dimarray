""" User-interface, should be relatively stable
"""
from axes import Axis, Axes
from core import Dimarray, array, align_axes, align_dims, broadcast_arrays
from collect import Dataset

try:
    from ncio import read, summary
    _ncio = True
except:
    raise
    _ncio = False
    print "ncio not imported"
    pass
