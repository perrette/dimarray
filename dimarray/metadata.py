""" Module to distinguish metadata from other variables, 
which makes it easier for netCDF I/O and printing to screen.
"""
import numpy as np
import copy
from collection import OrderedDict as odict

# 
# Attribute Descriptor
#

class Variable(object):
    """ Variable with metadata
    
    Help maintain an ordered list or attributes distinguishable from other 
    attributes. Typical metadata are:

    - units: variable units
    - descr: variable description (not including geo-location)
    - stamp: string information indicating e.g. coordinates or time stamp or history

    How it works: anything that is not private (does not start with "_"), is
    not in the "_metadata_exclude" list, and whose type is in "_metadata_types"
    is just considered a metadata and its name is added to "_metadata", so that
    1) the list is permanently up-to-date and 2) it is ordered (which is not the 
    case for self.__dict__)
    """
    # Most important field (e.g. ("values",) or ("values", "axes"))
    _metadata_exclude = () # variables which are NOT metadata

    # impose restriction about what type metadata can have
    # for now use python immutable types, eventually could authorize
    # everything which is netCDF writable
    _metadata_types = int, long, float, tuple, str, bool  # autorized metadata types

    def __init__(self):
	"""
	"""
	_metadata = odict() # ordered dictionary of metadata

    def __getattr__(self, att):
	""" set attributes: distinguish between metadata and other 
	"""
	# Searching first in __dict__ is the default behaviour, 
	# included here for clarity
	if att in self.__dict__:
	    return super(Variable, self).__getattr__(att)

	# check if att is a metadata
	elif att in self._metadata
	    return self._metadata[att]

	# again here, to raise a typical error message
	else:
	    return super(Variable, self).__getattr__(att)

    def __setattr__(self, att, val):
	""" set attributes: some are metadata, other not
	"""
	# add name to odict of metadata
	if self._ismetadata(att, val):
	    self._metadata[att] = val

	else:
	    super(Variable, self).__setattr__(att, val)

    def __detattr__(self, att):
	""" delete an attribute
	"""
	# default behaviour
	if att in self.__dict__:
	    super(Variable, self).__delattr__(att) # standard method

	# delete metadata if it applicable
	elif att in self._metadata:
	    del self._metadata[att]

	# again here, to raise a typical error message
	else:
	    super(Variable, self).__delattr__(att) # standard method


    def _ismetadata(self, att, val=None):
	""" returns True if att is metadata
	"""
	# if val is None, test if att in self._metadata
	# if att already in metadata, then "True"
	if val is None or att in self._metadata:
	    test = att in self._metadata

	# otherwise, check if it is a valid metadata
	else: 
	    test = not att.startswith('_') and att not in self._metadata_exclude \
		    and type(val) in self._metadata_types # also check type (to avoid limit errors)

	return test


    #
    # This methods are analogous to netCDF4 module in python, and useful
    # to set/get/det metadata when there is a conflict with another existing name
    #

    def ncattrs(self):
	return self._metadata.keys()

    def setncattr(self, att, val):
	""" force setting attribute to metadata
	"""
	if not type(val) in self._metadata_types:
	    raise TypeError("Found type {}. Only authorized metadata types: {}".format(type(val), self._metadata_types))

	self._metadata[att] = val

    def getncattr(self, att):
	""" 
	"""
	return self._metadata[att]

    def delncattr(self, att):
	""" 
	"""
	del self._metadata[att]

    #
    # "pretty" printing
    #

    def repr_meta(self):
	""" string representation for metadata
	"""
	return ", ".join(['{}: {}'.format(att, getattr(self, att)) for att in self.ncattrs()])


def test():
    import doctest
    import metadata
    doctest.testmod(metadata)

if __name__ == "__main__":
    test()
