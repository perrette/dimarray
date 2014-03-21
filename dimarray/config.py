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
indexing = "values"
indexing_broadcast = True

op_broadcast = True
op_reindex = True

display_max = 100 

def set_option(name, value):
    """ set global options
    """
    if name not in globals():
	raise ValueError("unknown option: {}".format(name))
    globals()[name] = value

def get_option(name):
    if name not in globals():
	raise ValueError("unknown option: {}".format(name))
    return globals()[name]
