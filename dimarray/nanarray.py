""" nanarray : treat NaNs as missing values

Added Behaviour: 
---------------

+ treats nans as missing values

Full Genealogy:
---------------

BaseArray => NanArray => LaxArray => DimArray

BaseArray: just emulate a numpy array
           + introduce the xs method (only one axis)
           + axis=0 by default (except in squeeze)
	   + `==` operator returns a boolean and not an array

NanArray : treats nans as missing values

LaxArray : add a name to each dimension 
	   + can give an axis name to `axis=`
	   + multidimensional slicing
	   + recursive `endomorphic` transformation on subsets of the whole space

DimArray : add values and metadata to each axis and to the array
	   + netCDF I/O
"""
import numpy as np
import copy
import basearray as ba
from tools import _NumpyDesc

class NanArray(ba.BaseArray):
    """ BaseArray which treats NaNs as missing values
    """
    def __init__(self, values, dtype=None, copy=False):
	""" instantiate a ndarray with values: note values must be numpy array

	values: first argument passed to np.array()
	dtype, copy: passed to np.array 
	"""
	self.values = np.array(values, dtype=dtype, copy=copy)

    def __repr__(self):
	""" screen printing
	"""
	header = "<NanArray>"
	return "\n".join([header,repr(self.values)])


    def apply(self, funcname, axis=None, skipna=True):
	""" apply numpy's `func` and return a BaseArray instance

	Generic description:

	input:
	    funcname : string name of the function to apply (must be a numpy method)
	    axis   : int or str (the axis label), or None [default 0] 
	    skipna : if True, skip the NaNs in the computation


	return:
	    result : from the same class (if axis is not None, else np.ndarray)

	Examples:
	--------
	>>> b = a.copy()
	>>> b[1] = np.nan
	>>> b
	<NanArray>
	array([[  0.,   1.,   2.,   3.,   4.,   5.],
	       [ nan,  nan,  nan,  nan,  nan,  nan],
	       [  2.,   3.,   4.,   5.,   6.,   7.],
	       [  3.,   4.,   5.,   6.,   7.,   8.],
	       [  4.,   5.,   6.,   7.,   8.,   9.]])
	>>> b.sum(axis=0)
	<NanArray>
	array([  9.,  13.,  17.,  21.,  25.,  29.])
	>>> b.values.sum(axis=0)
	array([ nan,  nan,  nan,  nan,  nan,  nan])
	"""
	assert type(funcname) is str, "can only provide function as a string"

	# Convert to MaskedArray if needed
	values = self.values
	if np.any(np.isnan(values)) and skipna:
	    values = np.ma.array(values, mask=np.isnan(values))

	# Apply numpy method
	result = getattr(values, funcname)(axis=axis) 

	# if scalar, just return it
	if not isinstance(result, np.ndarray):
	    return result

	# otherwise, fill NaNs back in
	return NanArray(result.filled(np.nan))

#    #
#    # Cumbersome update NanArray to BaseArray, for this small jump of little 
#    # use (could have replaced BaseArray with self.__class__) but this prepares
#    # bigger jumps when additional parameters are needed
#    #
#    def _operation(self, func, other):
#	result = super(NanArray, self)._operation(func, other)
#	return NanArray(result)
#
#    def __getitem__(self, item):
#	return NanArray(self.values[item])
#
#    def copy(self):
#	return NanArray(self.values.copy())
#
#    def squeeze(self, axis=None):
#	""" remove singleton dimensions
#	"""
#	result = super(NanArray, self).squeeze(func, axis=axis)
#	return NanArray(res)
#
#    def transpose(self, axes=None):
#	result = super(NanArray, self).transpose(axes=axes)
#	return NanArray(res)
#
#    def xs(self, *args, **kwargs):
#	""" see doc in BaseArray
#	"""
#	result = super(NanArray, xs).transpose(*args, **kwargs)
#	return NanArray(res)


array = NanArray # as convenience function


# 
# For testing
#
# CHECK README FORMATTING python setup.py --long-description | rst2html.py > output.html

def get_testdata():
    a = ba.get_testdata()
    return NanArray(a)

def _load_test_glob():
    """ return globs parameter for doctest.testmod
    """
    import doctest
    import numpy as np
    from nanarray import NanArray
    import nanarray as na

    a = get_testdata()
    values = a.values
    b = a[:2,:2] # small version

    return locals()

def test_doc(raise_on_error=False, globs={}, **kwargs):
    import doctest
    import nanarray as na

    kwargs['globs'] = _load_test_glob()

    return doctest.testmod(na, raise_on_error=raise_on_error, **kwargs)

if __name__ == "__main__":
    test_doc()
