class Config:
    """ Global configuration file

    Indexing/slicing option(s)
    - indexing: default indexing method

    Options that determine alignment behaviour during an operation:
    - op_broadcast: broadcast dimensions (alignment during an operation)
    - op_reindex: align all operands on a common axis (fill with NaN missing values)

    Display:
    - display_max: max number of array values displayed (if greater than that
		   array(...) will be shown)
    """
    op_broadcast=True
    op_reindex=True
    indexing="index"
    display_max = 100 

def set_option(name, value):
    """ set global options
    """
    setattr(Config, name, values)

def get_option(name):
    return getattr(Config, name)

