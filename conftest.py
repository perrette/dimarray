import dimarray as da
import pytest
import numpy as np

collect_ignore = ["setup.py", "docs/conf.py", "dist", "build", "tests/testing.py", "tests/test_mpl.py"]
collect_ignore.append('docs/_build_rst/dimarray.rst')

try:
    import netCDF4
except ImportError:
    # netCDF4 module is not present
    collect_ignore.append('dimarray/io/nc.py')

# test on python version before executing this file
#if sys.version_info[0] > 2:
#        collect_ignore.append("pkg/module_py2.py")


# nice dimarray display in error message
def pytest_assertrepr_compare(op, left, right):
    if (isinstance(left, da.DimArray) or isinstance(left, np.ndarray)) and (isinstance(right, da.DimArray) or isinstance(right, np.ndarray)):
    #and op == "==":

	return ['Comparing {} and {} instances:'.format(type(left).__name__, type(right).__name__)] \
		+ [''] \
		+ repr(left).split('\n') \
		+ ['']  \
		+ [' :: does not compare with :: ']  \
		+ ['']  \
		+ repr(right).split('\n')

## does not seem to work
## # provide fixture when running doctest (equivalent of doctest globs= parameter)
## @pytest.fixture(autouse=True)
## def np():
##     import numpy as np
##     return np
## 
## @pytest.fixture(autouse=True)
## def da():
##     import dimarray as da
##     return da
## 
## @pytest.fixture(autouse=True)
## def DimArray():
##     from dimarray import DimArray
##     return DimArray
## 
## @pytest.fixture(autouse=True)
## def Dataset():
##     from dimarray import Dataset
##     return Dataset
