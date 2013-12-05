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
from functools import partial

from shared import operation as _operation
import basearray as ba


class NanArray(ba.BaseArray):
    """ BaseArray which treats NaNs as missing values
    """

    def __init__(self, values, dtype=None, copy=False):
	""" instantiate a ndarray with values: note values must be numpy array

	values: first argument passed to np.array()
	dtype, copy: passed to np.array 
	"""
	self.values = np.array(values, dtype=dtype, copy=copy)

    @classmethod
    def _constructor(cls, values, names=None, dtype=None, copy=False):
	""" all transformations/operations call _constructor
	set default names to None
	"""
	return NanArray(values=values, names=names, dtype=dtype, copy=copy)

    #
    # Operations on array that require dealing with NaNs
    #

    def _ma(self, skipna=True):
	""" return a masked array in place of NaNs if skipna is True
	"""
	a = self.values

	# return numpy array is no NaN is present of if skipna is False
	if not np.any(np.isnan(a)) or not skipna:
	    return a

	masked_array = np.ma.array(a, mask=np.isnan(a), fill_value=np.nan)
	return masked_array

    @staticmethod
    def _array(res):
	""" reverse of _ma: get back standard numpy array, 

	filling nans back in if necessary
	"""
	# fills the nans back in
	if isinstance(res, np.ma.MaskedArray): 
	    res.fill(np.na)
	return res

    def apply(self, funcname, axis=0, skipna=True):
	""" apply `func` along an axis

	Generic description:

	input:
	    funcname : string name of the function to apply (must be a numpy method)
	    axis   : int or str (the axis label), or None [default 0] 
	    skipna : if True, skip the NaNs in the computation

	return:
	    result : from the same class (if axis is not None, else np.ndarray)

	Examples:
	--------

	>>> a.mean(axis='lon')
	dimensions(5,): lat
	array([ 2.5,  3.5,  4.5,  5.5,  6.5])

	Default to axis=0 (here "lat")

	>>> a.mean()
	dimensions(6,): lon
	array([ 2.,  3.,  4.,  5.,  6.,  7.])

	>>> a.mean(axis=None)
	4.5
	"""
	values = self._ma(skipna)

	# retrive bound method
	method = getattr(values, funcname)

	# return a float if axis is None
	if axis is None:
	    return method()

	axis = self._get_axis_idx(axis) 
	res_ma = method(axis=axis)
	res = self._array(res_ma) # fill nans back in if necessary

	if funcname.startswith("cum"):
	    names = self.names
	else:
	    names = self._get_reduced_names(axis) # name after reduction

	return self._constructor(res, names=names)

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
    from nanarray import NanArray
    import nanarray as na

    # same as basearray
    baglob = ba._load_test_glob()
    locals().update(baglob)

    a = get_testdata()
    values = a.values

    return locals()

def test_doc(raise_on_error=False, globs={}, **kwargs):
    import doctest
    import nanarray as na

    kwargs['globs'] = _load_test_glob()

    return doctest.testmod(na, raise_on_error=raise_on_error, **kwargs)

if __name__ == "__main__":
    test_doc()
