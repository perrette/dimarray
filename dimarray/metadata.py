""" This module contains describe the basic Variable type with values and metadata
"""
# 
# Attribute Descriptor
#
class Desc(object):
    """ Attribute descriptor to control access to a class's attribute
    via an `attrs` dictionary and getter/setter methods

    Example:

    >>> class A:
    ...     name = Desc(name="name", default="", atype=str)
    ...     units = Desc("units", "", str)
    ...     anything = Desc("anything")
    >>> a = A()
    >>> a.name
    ''
    >>> a.name = "mynane"
    >>> a.name
    "myname"
    >>> a._attrs
    {'name':'myname'}
    >>> del a.name
    >>> a.name
    ''
    """
    atype = str, int, long, bool, float, tuple

    def __init__(name, default="", atype=None):
	"""
	name : attribute name in obj._attrs
	default: default returned values if not set by user
	atype: attrubute type (can have multiple allowed types)
	"""
	self.name = name
	self.default = default

	# provide info on type
	if atype is None:
	    self.atype = tuple(atype)

    def __get__(self, obj, cls=None):
	""" access to attribute
	"""
	if self.name in obj._attrs:
	    val = obj._attrs[self.name]
	else:
	    val = self.default # default value

    def __set__(self, obj, val):
	""" access to attribute
	"""
	# check that attribute has the right type
	if type(val, self.atype):
	    raise TypeError("attribute {} must be {}, got {}".format(self.name, self.atype, type(val)))
	obj._attrs[self.name] = val

    def __del__(self):
	""" reset to default value
	"""
	del obj._attrs[self.name]


class Variable(object):
    """ A variable has values and four typical attributes:

    - values: stored as list or numpy array

    - name : variable name
    - units: variable units
    - descr: variable description (not including geo-location)
    - stamp: string information indicating e.g. coordinates or time stamp or history

    Also accessible the values' type

    TO DO: check with actual netCDF conventions and possibly update
    """
    _attrs = {}
    names = Desc("name", "", str)
    units = Desc("units", "", str)
    descr = Desc("descr", "", str)
    stamp = Desc("stamp", "", str)

    def ncattrs(self):
	""" to be written to netCDF as attribute (only non-empty attributes)
	"""
	basic = filter(lambda x: x and x in self._attrs, ["name","units","descr","stamp"])  # provide separately to keep it sorted (nicer display when printing)
	user =  [nm for nm in self._attrs if nm not in basic and not nm.startswith("_")]
	return basic + user

    def __setattr__(self, att, val):
	""" set non-standard attribute (via `.`)
	"""
	self._attrs[att] = val # we get there only if val not already defined above

    def __getattr__(self, att):
	""" retrieve non-standard attribute
	"""
	if att in self._attrs:
	    return self._attrs[att]

	# This will never happen with a standard attribute as defined above
	else:
	    raise ValueError("Attribute not found: "+att)

    @property
    def size(self): 
	return np.size(self.values)

    @property
    def dtype(self): 
	return _convert_dtype(np.array(self.values).dtype)

    @property
    def __array__(self): 
	return self.values.__array__


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
