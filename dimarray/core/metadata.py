""" Help handling of metadata

Conventions:
    __metadata_exclude__ : define a list of object attributes which are no metadata
        (e.g. values, or axes, if they are no descriptors)
    __metadata_include__ : to include elements starting with '_' or not 
        appearing in an instance dict (e.g. defined via descriptors)
"""
import warnings 
import copy
import json
from collections import OrderedDict as odict

# 
# Determine what is a metadata
#
def _keys(obj):
    """ returns metadata variable names of a MetadataBase instance
    """
    candidates = obj.__dict__.keys()
    #candidates.extend(obj.__metadata_include__) # for class attributes which are not in obj __dict__
    #candidates = set(candidates) # remove doubles
    test = lambda k : k in obj.__metadata_include__ or \
            (k not in obj.__metadata_exclude__ \
            #and (not hasattr(obj.__class__, k)) \ # remove that one, need explicit exclusion via __metadata_exclude__
            and (not k.startswith('_')))
            # do not include class attributes (even modified) as attribute
            # unless explicitly specified

    keys = [k for k in candidates if hasattr(obj, k) and test(k)]

    # check for additional metadata 
    if not hasattr(obj, "_metadata_private"):
        obj._metadata_private = {} # initialize

    keys.extend(obj._metadata_private.keys())

    # check...
    if len(set(keys) ) != len(keys):
        warnings.warn("metadata present both in __dict__ and in _metadata_private !")
        #1/0
        # will be accessed and destroyed first from metadata

    return keys


#
# methods how to access an element
#
def _set_metadata(obj, k, val, warn=True):
    """ set one metadata
    """
    # make sure _metadata_private attribute exists
    _check_private(obj)

    is_class_attribute = hasattr(obj.__class__, k)
    is_class_method = is_class_attribute and callable(getattr(obj, k))

    if is_class_method \
            or k in obj.__metadata_exclude__ \
            or ' ' in k \
            or (k.startswith('_') and not k in obj.__metadata_include__): 
        if warn : 
            warnings.warn("{} is private or protected name, store in _metadata_private".format(k))
        obj._metadata_private[k] = val

    else:
        try:
            setattr(obj, k, val)
        except AttributeError:
            if warn:
                warnings.warn("{} is private or protected name, store in _metadata_private".format(k))
            obj._metadata_private[k] = val

    return

def _get_metadata(obj, k):
    """ get one metadata
    """
    _check_private(obj)
    if k in obj._metadata_private:
        return obj._metadata_private[k]
    elif hasattr(obj, k):
        return getattr(obj, k)
    else:
        raise ValueError("Metadata not found: {}".format(k))

def _del_metadata(obj, k):
    _check_private(obj)
    if k in obj._metadata_private:
        del obj._metadata_private
    elif hasattr(obj, k):
        if k in obj.__dict__:
            delattr(obj, k)
            # otherwise class attribute, won't work
    else:
        raise ValueError("Metadata not found: {}".format(k))

def _check_private(obj):
    """ add _metadata_private attribute
    """
    if not hasattr(obj, "_metadata_private"):
        obj._metadata_private = {} # initialize

    
# Function to set/get all metadata
def _metadata(obj, metadata=None):
    """ if metadata is provided, set 
    otherwise, get metadata.
    """
    # return metadata
    if metadata is None:
        return {k:_get_metadata(obj, k) for k in _keys(obj)}

    # otherwise set metadata
    assert isinstance(metadata, dict), "metadata can only be a dictionary"

    # first delete everything
    for k in _keys(obj):
        _del_metadata(obj, k)

    # then set new metadata
    for k in metadata:
        _set_metadata(obj, k, metadata[k])

#
# Temporary !!
#
class _Metadata(dict):
    """ dict instance used to help transition
    between current dictionary access and future 
    method-like (call) access in next versions.
    """
    def __init__(self, obj): #*args, **kwargs):
        super(_Metadata, self).__init__(_metadata(obj))
        self.obj = obj

    def __getitem__(self, key):
        #warnings.warn("Please use _metadata() as a method.")
        warnings.warn("Please use _metadata() as a method.",DeprecationWarning)
        return self[key]

    def __setitem__(self, key, value):
        warnings.warn("Please use set_metadata() method.",DeprecationWarning)
        return _set_metadata(self.obj, key, value)

    def __delitem__(self, key):
        warnings.warn("Please use del_metadata() method.",DeprecationWarning)
        return _det_metadata(self.obj, key)

    def __repr__(self):
        warnings.warn("Please use _metadata() as a method.",DeprecationWarning)
        return dict.__repr__(self)
        #return super(_Metadata, self).__repr__()

    def __call__(self, metadata=None):
        if metadata is None:
            return dict(self) # return a proper dictionary

        else:
            return _metadata(self.obj, metadata)

class MetadataBase(object):
    """ It contains a _metadata method that check for all 
    instance attributes which are not private. This behaviour
    can be modified by the class attributes 
    __metadata_exclude__ and __metadata_include__.

    >>> class A(MetadataBase):
    ...     __metadata_exclude__ = ['b']
    ...     __metadata_include__ = ['e','_f']
    ...     c = 3
    ...     d = 7
    ...     e = 8
    >>> a = A()
    >>> a.a = 2
    >>> a.b = 3
    >>> a.c = 4
    >>> a._f = 5
    >>> a._g = 6
    >>> a._metadata()
    {'a': 2, '_f': 5, 'c': 4}
    >>> a._metadata({'a':45,'_f':22,'b':11}) # set new attribute
    >>> a._metadata()
    {'a': 45, '_f': 22, 'b': 11}
    >>> a._f
    22
    """
    __metadata_exclude__ = []  # instance dict that do not show up
    __metadata_include__ = []  # show up even though it is private

    # _metadata will become a function
    # for now a python trick is used to keep allowing 
    # both dict-like and function-like access.
    @property
    def _metadata(self):
        return _Metadata(self)

    @_metadata.setter
    def _metadata(self, metadata):
        warnings.warn("Please use _metadata() as a method.",DeprecationWarning)
        _metadata(self, metadata)

    def _metadata_summary(self):
        return "\n".join([" "*8+"{} : {}".format(key, value) for key, value  in self._metadata.iteritems()])
        #return repr(self._metadata())

    def summary(self):
        """ summary string representation (with metadata)
        """
        line = repr(self)
        line += "\n"+self._metadata_summary()
        return line

    def set_metadata(self, key, value):
        """ set single metadata, with name check
        """
        _set_metadata(self, key, value, warn=False)

    def get_metadata(self, key):
        """ get single metadata, with name check
        """
        return _get_metadata(self, key)

    def del_metadata(self, key):
        """ delete single metadata, with name check
        """
        _del_metadata(self, key)

#class Metadata(object):
#    """ Variable with metadata
#    
#    Help maintain an ordered list or attributes distinguishable from other 
#    attributes. Typical metadata are:
#
#    - units: variable units
#    - descr: variable description (not including geo-location)
#    - stamp: string information indicating e.g. coordinates or time stamp or history
#
#    How it works: anything that is not private (does not start with "_"), is
#    not in the "_metadata_exclude" list, and whose type is in "_metadata_types"
#    is just considered a metadata and its name is added to "_metadata", so that
#    1) the list is permanently up-to-date and 2) it is ordered (which is not the 
#    case for self.__dict__)
#
#    >>> v = Metadata()
#    >>> v.yo = 123
#    >>> v.name = 'myname'
#    >>> v.ya = 'hio'
#    >>> v._summary_meta()
#    'yo: 123, name: myname, ya: hio'
#    """
#    # Most important field (e.g. ("values",) or ("values", "axes"))
#    _metadata_exclude = () # variables which are NOT metadata
#
#    # impose restriction about what type metadata can have
#    # for now use python immutable types, eventually could authorize
#    # everything which is netCDF writable
#    _metadata_types = int, long, float, tuple, str, bool, unicode # autorized metadata types
#
#    _metadata_transform = True # conserve metadata via axis transform?
#    _metadata_units = "units" # metadata which follow unit-like rules in operations
#    _metadata_operation_warning = True
#
#    def __init__(self):
#        """
#        """
#        object.__setattr__(self, "_metadata", odict()) # ordered dictionary of metadata
#
#    def __getattr__(self, att):
#        """ set attributes: distinguish between metadata and other 
#        """
#        # Searching first in __dict__ is the default behaviour, 
#        # included here for clarity
#        if att in self.__dict__:
#            return self.__getattribute__(att) # no __getattr__ in object
#
#        # check if att is a metadata
#        elif att in self._metadata:
#            return self._metadata[att]
#
#        #elif: or att.startswith('_'):
#
#        # again here, to raise a typical error message
#        else:
#            return self.__getattribute__(att)
#
#    def __setattr__(self, att, val):
#        """ set attributes: some are metadata, other not
#        """
#        # add name to odict of metadata
#        if self._ismetadata(att, val):
#            self._metadata[att] = val
#
#        else:
#            object.__setattr__(self, att, val)
#
#    def __detattr__(self, att):
#        """ delete an attribute
#        """
#        # default behaviour
#        if att in self.__dict__:
#            object.__delattr__(self, att) # standard method
#
#        # delete metadata if it applicable
#        elif att in self._metadata:
#            del self._metadata[att]
#
#        # again here, to raise a typical error message
#        else:
#            object.__delattr__(self, att) # standard method
#
#
#    def _ismetadata(self, att, val=None):
#        """ returns True if att is metadata
#        """
#        # if val is None, test if att in self._metadata
#        # if att already in metadata, then "True"
#        if val is None or att in self._metadata:
#            test = att in self._metadata
#
#        # otherwise, check if it is a valid metadata
#        else: 
#            test = not att.startswith('_') and att not in self._metadata_exclude \
#                    and (type(val) in self._metadata_types or np.isscalar(val))# also check type (to avoid limit errors)
#
#        return test
#
#
#    #
#    # This methods are analogous to netCDF4 module in python, and useful
#    # to set/get/det metadata when there is a conflict with another existing name
#    #
#
#    def ncattrs(self):
#        return self._metadata.keys()
#
#    def setncattr(self, att, val):
#        """ force setting attribute to metadata
#        """
#        if not np.isscalar(val) and not type(val) in self._metadata_types:
#            raise TypeError("Found type {}. Only authorized metadata types: {}".format(type(val), self._metadata_types))
#
#        self._metadata[att] = val
#
#    def getncattr(self, att):
#        """ 
#        """
#        return self._metadata[att]
#
#    def delncattr(self, att):
#        """ 
#        """
#        del self._metadata[att]
#
#    #
#    # "pretty" printing
#    #
#
#    def _metadata_summary(self):
#        """ string representation for metadata
#        """
#        return ", ".join(['{}: {}'.format(att, self.getncattr(att)) for att in self.ncattrs()])
#
#    def summary(self):
#        """ summary string representation (with metadata)
#        """
#        line = repr(self)
#        line += "\n"+self._metadata_summary()
#        return line
#
#    #
#    # add a new stamp to "stamp" list via json 
#    #
#    def _metadata_stamp(self, stamp=None):
#        """ append a new transform or operation stamp to metadata
#        """
#        if stamp:
#            append_stamp(self._metadata, stamp, inplace=True)
#
#def append_stamp(metadata, stamp, inplace=False):
#    """ append a new transform or operation stamp to metadata
#    """
#    if not inplace:
#        metadata = metadata.copy()
#
#    # add transform (json-ed list)
#    if "stamp" not in metadata:
#        metadata["stamp"] = json.dumps(()) # json-ed empty tuple
#
#    # de-json the stamp
#    stamp_tuple = json.loads(metadata['stamp'])
#
#    # append the new stamp
#    stamp_tuple += (stamp,)
#
#    # transform back to string
#    metadata['stamp'] = json.dumps(stamp_tuple)
#
#    if not inplace:
#        return metadata
#
#def test(**kwargs):
#    import doctest
#    import metadata
#    doctest.testmod(metadata, **kwargs)
#
#if __name__ == "__main__":
#    test()
