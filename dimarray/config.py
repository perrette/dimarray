""" Global configuration file

Indexing/slicing option(s)
- indexing: default indexing method 
- indexing_broadcast: determines if array indices should be 
    - broadcast onto a single axis (True, the default)
    - or be dealt with independently from each other (False)

Options that determine alignment behaviour during an operation:
- op_broadcast: broadcast dimensions (alignment during an operation)
- op_reindex: align all operands on a common axis (fill with NaN missing values)

Display:
- display_max: max number of array values displayed (if greater than that
               array(...) will be shown)
"""
from collections import OrderedDict as odict

class Params(odict):
    """ class to make user interface easier, e.g. displays all parameter values
    """
    def __repr__(self): 
        """ nicer printing to string
        """
        return "\n".join(["{:>20} = {:<20}".format(k,self[k]) for k in self])

rcParams = Params()
rcParamsHelp = Params() # help

# Define options

# indexing
rcParams['indexing.by'] = "values"
rcParamsHelp['indexing.by'] = "'values' or 'position' (default: 'values'). Examples: ('values') 2014 or 'red'; ('position') 0, 1, 2 ..., The `.ix` attribute is a toogle between both modes.\n"

rcParams['indexing.broadcast'] = True
rcParamsHelp['indexing.broadcast'] = "bool (default: True). If True, broadcast array indices to collapse indexed dimensions into single shape, like numpy arrays, otherwise (if False) like matlab or pandas, multi-dimensional indexing treated as chained call, with dimensions independent from each other: such behaviour is normally obtained via `a.box[...]` instead of `a[...]`.\n"

# options on operations
rcParams['op.broadcast'] = True
rcParamsHelp['op.broadcast'] = "bool (default: True). If True, broadcast arrays before an operation, according to dimension names, otherwise (if False) basic numpy rules, which may involve broadcasting in simple cases\n"

rcParams['op.reindex'] = True
rcParamsHelp['op.reindex'] = "bool (default True). If True, reindex axis before an operation, such as to match two time series. Otherwise (if False), size and indices must exactly match.\n"

# options for display
rcParams['display.max'] = 100 
rcParamsHelp['display.max'] = "int (default: 100): max array size shown. Note: if set to large values (e.g. inf), follows numpy display.\n"

def set_option(name, value):
    """ set global options
    """
    if name not in rcParams:
        raise ValueError("unknown option: {}".format(name))
    rcParams[name] = value

def get_option(name):
    if name not in rcParams:
        raise ValueError("unknown option: {}".format(name))
    return rcParams[name]

def print_options():
    print "Options description:"
    print rcParamsHelp
    print "\nOptions values:"
    print rcParams
