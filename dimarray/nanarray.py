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

from tools import _operation, _NumpyDesc
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
    def _constructor(cls, values, dtype=None, copy=False):
	""" 
	"""
	return NanArray(values=values, dtype=dtype, copy=copy)

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

    def apply(self, funcname, skipna=True, args=(), **kwargs):
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
	>>> b[2] = 0.
	>>> res = b.sum()
	>>> b[2] = np.nan
	>>> b.sum()
	>>> b == c
	True
	"""
	values = self._ma(skipna)

	# retrive bound method
	method = getattr(values, funcname)

	# return a float if axis is None
	if axis is None:
	    return method()

	res_ma = method(axis=axis)
	res = self._array(res_ma) # fill nans back in if necessary

	return self._constructor(res)
	assert type(funcname) is str, "can only provide function as a string"

	# retrieve bound method
	method = getattr(self.values, funcname)

	# use None as default values otherwise this would not work !
	# as axis can be in *args or **kwargs
	# or one would need to inspect method's argument etc...


	## return a float if axis is None, just as numpy does
	#if 'axis' in kwargs and kwargs['axis'] is None:
	#    kwargs['axis'] = 0 # default values

	values = method(*args, **kwargs)

	# check wether the result is a scalar, if yes, export
	values, test_scalar = _check_scalar(values)

	if test_scalar:
	    return values
	else:
	    return self._constructor(values)

    #
    # Add numpy transforms
    #
    mean = _NumpyDesc("apply", "mean")
    median = _NumpyDesc("apply", "median")
    sum  = _NumpyDesc("apply", "sum")
    diff = _NumpyDesc("apply", "diff")
    prod = _NumpyDesc("apply", "prod")
    min = _NumpyDesc("apply", "min")
    max = _NumpyDesc("apply", "max")
    ptp = _NumpyDesc("apply", "ptp")
    cumsum = _NumpyDesc("apply", "cumsum")
    cumprod = _NumpyDesc("apply", "cumprod")

    take = _NumpyDesc("apply", "take")

    def take(self, indices, axis=0):
	""" apply along-axis numpy method
	"""
	res = self.values.take(indices, axis=axis)
	return self._constructor(res)

    def transpose(self, axes=None):
	""" apply along-axis numpy method
	"""
	res = self.values.take(indices, axes)
	return self._constructor(res)

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
