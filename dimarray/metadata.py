""" This module contains describe the basic Variable type with values and metadata

NOTE: experimental dealing with decriptor, will be removed if it is more 
confusing than it brings benefits (could also use @property decorators)
"""
from collections import OrderedDict as odict
import numpy as np
import copy

# 
# Attribute Descriptor
#
class Metadata(object):
    """ Metadata descriptor to control access to a class's attribute
    via an `_metadata` dictionary and __get__/__set__ methods

    Advantages vs basic `.` attribute:
	- make the distinction between data and metadata: e.g. print and I/O
	- fine control of default vs user-defined attributes
	- move unclear things out of the way (__getattr__ etc...) in actual class

    NOTE: this is experimental and might be removed if it 
	  provdes too confusing for standard usage.

    Example:

    >>> class A(object):
    ...     _metadata = dict() # can be ordered
    ...     name = Metadata(name="name", default="", atype=str)
    ...     units = Metadata("units", "", str)
    ...     anything = Metadata("anything")
    >>> a = A()
    >>> a.units  # units not provided
    ''
    >>> a._metadata  
    {}
    >>> a.units = "some units"
    >>> a.units     # some units provided
    'some units'
    >>> a._metadata
    {'units': 'some units'}
    >>> a.units= "" # provided as unitless
    >>> a._metadata  
    {'units': ''}
    >>> del a.units
    >>> a.units
    ''
    """
    #atype = str, int, long, bool, float, tuple

    def __init__(self, name, default="", atype=None):
	"""
	name : attribute name in obj._metadata
	default: default returned values if not set by user
	atype: attrubute type (can have multiple allowed types)
	"""
	self.name = name
	self.default = default
	if atype is None:
	    atype = type(default)
	if type(atype) is type:
	    atype = (atype,)
	self.atype = atype

    def __get__(self, obj, cls=None):
	""" access to attribute: return default if not user-defined !
	"""
	if self.name in obj._metadata:
	    val = obj._metadata[self.name]
	else:
	    val = self.default # default value
	return val

    def __set__(self, obj, val):
	""" access to attribute
	"""
	# check that attribute has the right type
	if not type(val) in self.atype:
	    raise TypeError("attribute {} must be {}, got {}".format(self.name, self.atype, type(val)))

	obj._metadata[self.name] = val

    def __delete__(self, obj):
	""" reset to default value
	"""
	del obj._metadata[self.name]


class Variable(object):
    """ A variable has values and metadata

    - values: stored as list or numpy array under _values

    Typical metadata are:

    - name : variable name
    - units: variable units
    - descr: variable description (not including geo-location)
    - stamp: string information indicating e.g. coordinates or time stamp or history

    They are stored under _metadata attributes. It is possible to set default attributes
    using the metadata descriptor and access is 

    TO DO: check with actual netCDF conventions 

    Examples:
    ---------

    >>> v = Variable([1,2])
    >>> v.name
    ''
    >>> v.name = "myname"
    >>> v.name
    'myname'
    >>>
    """
    #
    # Properties
    #
    name  = Metadata("name", "", str)
    units = Metadata("units", "", str)
    descr = Metadata("descr", "", str)
    stamp = Metadata("stamp", "", str)

    # store attributes in an ordered dict (to be copied at initialization)
    _metadata_default = odict() 

    # Set values as a property for enhanced control
    @property 
    def values(self):
	return self._values

    @values.setter
    def values(self, newval):
	if np.shape(newval) != np.shape(self._values):
	    msg = "To reshape dimensions try one of the exising methods "
	    msg += "(squeeze, transpose...) or create a new <{cls}> instance".format(cls=self.__class__.__name__)
	    raise ValueError(msg)

    #
    # ...
    #

    def __init__(self, values, **attrs):
	""" typical initialization call
	"""
	self._values = values
	self._metadata = self._metadata_default.copy()
	self.set(inplace=True, **attrs)

    def __getattr__(self, att):
	""" return attributes
	"""
	if att in self._metadata:
	    return self._metadata[att]

	else:
	    raise AttributeError("unknown attribute "+att)

    def ncattrs(self):
	""" to be written to netCDF as attribute (only non-empty attributes)
	"""
	return self._metadata.keys()

    @property
    def size(self): 
	return np.size(self.values)

    @property
    def dtype(self): 
	return _convert_dtype(np.array(self.values).dtype)

    @property
    def __array__(self): 
	return self.values.__array__

    def copy(self):
	""" shallow copy
	"""
	return copy.copy(self)

    def set(self, inplace=False, **kwargs):
	""" 
	inplace: modify attributes in-place, return None (useful to respect closure)

	a.set(units=m).plot() # in case units was forgotten
	a.set(_slicing="numpy")[:30]
	a.set(_slicing="exact")[1971.42]
	a.set(_slicing="nearest")[1971]
	a.set(name="myname", inplace=True) # modify attributes inplace
	"""
	if inplace: obj = self
	else: obj = self.copy()

	for k in kwargs:
	    if hasattr(self, k) or k.startswith('_'):
		setattr(self, k, kwargs[k])
	    else:
		print "add to metadata"
		self._metadata[k] = kwargs[k]

	if not inplace:
	    return obj

    def repr_meta(self):
	""" string representation for metadata
	"""
	return ", ".join(['%s: "%s"' % (att, getattr(self, att)) for att in self.ncattrs()])

def _convert_dtype(dtype):
    """ convert numpy type in a python type
    """
    if dtype is np.dtype(int):
	type_ = int

    elif dtype in [np.dtype('S'), np.dtype('S1')]:
	type_ = str

    else:
	type_ = float

    return type_

def test():
    import doctest
    import metadata
    doctest.testmod(metadata)

if __name__ == "__main__":
    test()
