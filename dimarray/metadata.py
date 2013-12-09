""" Module to distinguish metadata from other variables, 
which makes it easier for netCDF I/O and printing to screen.
"""
import numpy as np
import copy
import json
from units import Units

from collections import OrderedDict as odict

# 
# Attribute Descriptor
#

class Metadata(object):
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

    _metadata_transform = True # conserve metadata via axis transform?
    _metadata_units = "units" # metadata which follow unit-like rules in operations
    _metadata_operation_warning = True

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
	    return super(Metadata, self).__getattr__(att)

	# check if att is a metadata
	elif att in self._metadata:
	    return self._metadata[att]

	# again here, to raise a typical error message
	else:
	    return super(Metadata, self).__getattr__(att)

    def __setattr__(self, att, val):
	""" set attributes: some are metadata, other not
	"""
	# add name to odict of metadata
	if self._ismetadata(att, val):
	    self._metadata[att] = val

	else:
	    super(Metadata, self).__setattr__(att, val)

    def __detattr__(self, att):
	""" delete an attribute
	"""
	# default behaviour
	if att in self.__dict__:
	    super(Metadata, self).__delattr__(att) # standard method

	# delete metadata if it applicable
	elif att in self._metadata:
	    del self._metadata[att]

	# again here, to raise a typical error message
	else:
	    super(Metadata, self).__delattr__(att) # standard method


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

    #
    # add a new stamp to "stamp" list via json 
    #
    def _metadata_append_stamp(self, stamp):
	""" append a new transform or operation stamp to metadata
	"""
	# add transform (json-ed list)
	if "stamp" not in self._metadata:
	    self._metadata["stamp"] = json.dumps([]) # json-ed empty list

	# de-json the stamp
	stamp_list = json.loads(self._metadata['stamp'])

	# append the new stamp
	stamp_list.append(stamp)

	# transform back to string
	self._metadata['stamp'] = json.dumps(stamp_list)

    #
    # update metadata after an axis transform
    #
    def _metadata_update_transform(self, other, transform, axis):
	""" 
	other: Metadata instance
	transform: transformation name
	axis	: axis name
	"""
	# first copy all other metadata if applicable
	if self._metadata_transform:
	    self._metadata = other._metadata # this includes "stamp"

	# new stamp
	axis = other.axes[axis]
	d = dict(transform=transform, axis=axis.name, start=axis.values[0], end=axis.values[-1])
	stamp = "{transform}({axis}={start}:{end})".format(**d)

	self._metadata_append_stamp(stamp)

    #
    # update metadata after an operation
    #
    def _metadata_update_operation(self, o1, other, op):
	"""
	"""
	return # later

	if not "units" in o1._metadata:
	    return

	#
	# converves for add/subtract
	#

	if op in ("add","subtract"):
	    # do nothing if different dimensions originally (could raise a warning..)
	    if units1 != o2._metadata["units"]:
		if _metadata_operation_warning:
		    raise Warning("different units by add/subtract")
	    self._metadata["units"] = o1._metadata["units"]
	    return

	# convert first operand units
	try:
	    units1 = o1._metadata["units"]
	    units1 = Units.loads(units1)

	except Exception, msg:
	    print msg
	    print "failed to convert units", units1
	    return 

	if op == "power":
	    units = units1.power(other)
	    self._metadata["units"] = units.dumps()
	    return 

	# convert second operand units
	try:
	    units2 = other._metadata["units"]
	    units2 = Units.loads(units2)

	except Exception, msg:
	    print msg
	    print "failed to convert units", units2
	    return 

	# combine
	if op == "multiply":
	    new = units1.multiply(units2)
	    self._metadata["units"] = new.dumps()

	elif op == "divide":
	    new = units1.divide(units2)
	    self._metadata["units"] = new.dumps()

	else:
	    print "unknown unit operation",op

	return


def test():
    import doctest
    import metadata
    doctest.testmod(metadata)

if __name__ == "__main__":
    test()
