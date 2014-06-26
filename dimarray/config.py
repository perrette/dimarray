""" Global configuration file

See rcParams for the options and rcParamsHelp for their meaning.
dimarray.print_options() provide an overview.
"""
from collections import OrderedDict as odict

class Params(odict):
    """ class to make user interface easier, e.g. displays all parameter values
    """
    def __repr__(self): 
        """ nicer printing to string
        """
        lines = []
        for k in self:
            val = self[k]
            if isinstance(val, str):
                val = "'{}'".format(val)
            elif type(val) is bool:
                val = "{}".format(val)
            line = "{:>20} = {:<}".format(k,val)
            lines.append(line)

        return "\n".join(lines)
        #return "\n".join(["{:>20} = {}".format(k,self[k]) for k in self])

rcParams = Params()
rcParamsHelp = Params() # help

# indexing
rcParams['indexing.by'] = "label"
rcParamsHelp['indexing.by'] = "'label' or 'position' (default: 'label'). Examples: ('label') 2014 or 'red'; ('position') 0, 1, 2 ..., The `.ix` attribute is a toogle between both modes.\n"

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

# options for optimization
rcParams['optim.use_pandas'] = True
rcParamsHelp['optim.use_pandas'] = 'bool (default: True). If True, use pandas for indexing by default'


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
