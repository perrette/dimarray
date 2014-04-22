""" Module to distinguish metadata from other variables, 
which makes it easier for netCDF I/O and printing to screen.
"""
import numpy as np
import copy
import json

from collections import OrderedDict as odict

# 
# Attribute Descriptor
#
class MetadataDesc(object):
    """ descriptor to set/get metadata

    >>> class A(object):
    ...            _metadata = MetadataDesc(exclude=('b',))
    ...            c = 3
    >>> a = A()
    >>> a.b = 4
    >>> a.c = 5
    >>> a._metadata
    {'c': 5}
    """ 
    def __init__(self, exclude=None, types=None):
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            else:
                assert type(exclude) in (list, tuple)
        else:
            exclude = []

        #if types is None:
        #    types = int, long, float, tuple, str, bool, unicode # autorized metadata types

        self.types = types
        self.exclude = exclude

    def __get__(self, obj, cls=None):
        """ just get 
        """ 
        return {k:getattr(obj, k) for k in obj.__dict__ if k not in self.exclude and not k.startswith('_')}

    def __set__(self, obj, meta):
        """
        """
        assert isinstance(meta, dict), "metadata can only be a dictionary"
        for k in meta:
            val = meta[k]
            if self.types is not None and not np.isscalar(val) and not type(val) in self.types:
                raise TypeError("Got metadata type {}. Only authorized metadata types: {}".format(val.__class__.__name__, [t.__name__ for t in self.types]))
            setattr(obj, k, val)


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

    >>> v = Metadata()
    >>> v.yo = 123
    >>> v.name = 'myname'
    >>> v.ya = 'hio'
    >>> v.repr_meta()
    'yo: 123, name: myname, ya: hio'
    """
    # Most important field (e.g. ("values",) or ("values", "axes"))
    _metadata_exclude = () # variables which are NOT metadata

    # impose restriction about what type metadata can have
    # for now use python immutable types, eventually could authorize
    # everything which is netCDF writable
    _metadata_types = int, long, float, tuple, str, bool, unicode # autorized metadata types

    _metadata_transform = True # conserve metadata via axis transform?
    _metadata_units = "units" # metadata which follow unit-like rules in operations
    _metadata_operation_warning = True

    def __init__(self):
        """
        """
        object.__setattr__(self, "_metadata", odict()) # ordered dictionary of metadata

    def __getattr__(self, att):
        """ set attributes: distinguish between metadata and other 
        """
        # Searching first in __dict__ is the default behaviour, 
        # included here for clarity
        if att in self.__dict__:
            return self.__getattribute__(att) # no __getattr__ in object

        # check if att is a metadata
        elif att in self._metadata:
            return self._metadata[att]

        #elif: or att.startswith('_'):

        # again here, to raise a typical error message
        else:
            return self.__getattribute__(att)

    def __setattr__(self, att, val):
        """ set attributes: some are metadata, other not
        """
        # add name to odict of metadata
        if self._ismetadata(att, val):
            self._metadata[att] = val

        else:
            object.__setattr__(self, att, val)

    def __detattr__(self, att):
        """ delete an attribute
        """
        # default behaviour
        if att in self.__dict__:
            object.__delattr__(self, att) # standard method

        # delete metadata if it applicable
        elif att in self._metadata:
            del self._metadata[att]

        # again here, to raise a typical error message
        else:
            object.__delattr__(self, att) # standard method


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
                    and (type(val) in self._metadata_types or np.isscalar(val))# also check type (to avoid limit errors)

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
        if not np.isscalar(val) and not type(val) in self._metadata_types:
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
        return ", ".join(['{}: {}'.format(att, self.getncattr(att)) for att in self.ncattrs()])

    #
    # add a new stamp to "stamp" list via json 
    #
    def _metadata_stamp(self, stamp=None):
        """ append a new transform or operation stamp to metadata
        """
        if stamp:
            append_stamp(self._metadata, stamp, inplace=True)

def append_stamp(metadata, stamp, inplace=False):
    """ append a new transform or operation stamp to metadata
    """
    if not inplace:
        metadata = metadata.copy()

    # add transform (json-ed list)
    if "stamp" not in metadata:
        metadata["stamp"] = json.dumps(()) # json-ed empty tuple

    # de-json the stamp
    stamp_tuple = json.loads(metadata['stamp'])

    # append the new stamp
    stamp_tuple += (stamp,)

    # transform back to string
    metadata['stamp'] = json.dumps(stamp_tuple)

    if not inplace:
        return metadata

def test(**kwargs):
    import doctest
    import metadata
    doctest.testmod(metadata, **kwargs)

if __name__ == "__main__":
    test()
