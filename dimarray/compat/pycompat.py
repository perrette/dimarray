import sys
import itertools

PY3 = sys.version_info[0] >= 3

if PY3:
    basestring = str
    unicode_type = str
    bytes_type = bytes
    def dictitems(d):
        return list(d.items())
    def dictkeys(d):
        return list(d.keys())
    def dictvalues(d):
        return list(d.values())
    range = range
    zip = zip
    from collections import OrderedDict
else:
    # Python 2
    basestring = basestring
    unicode_type = unicode
    bytes_type = str
    def dictitems(d):
        return d.items()
    def dictkeys(d):
        return d.keys()
    def dictvalues(d):
        return d.values()
    zip = itertools.izip
    range = xrange
    from itertools import izip as zip
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict
