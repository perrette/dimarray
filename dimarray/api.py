""" User-interface, should be relatively stable
"""
from axes import Axis, Axes
from core import Dimarray, array
from collect import Dataset

try:
    from ncio import read, write, summary
    _ncio = True
except:
    _ncio = False
    print "ncio not imported"
    pass
