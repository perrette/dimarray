import dimarray as da
import pytest
import numpy as np
import warnings

collect_ignore = ["setup.py", "docs/conf.py", "dist", "build", "tests/testing.py", "tests/test_mpl.py", "docs/scripts"]
collect_ignore.append('docs/_build_rst/dimarray.rst')

try:
    import netCDF4
except ImportError:
    # netCDF4 module is not present
    warnings.warn("netCDF4 cannot be imported, skip netCDF4 tests and much of the documentation")
    collect_ignore.extend(['dimarray/io/nc.py', 'tests/test_nc.py'])
    collect_ignore.extend(['docs/_notebooks_rst/netcdf.rst',
                           'docs/_notebooks_rst/tutorial.rst',
                           'docs/_notebooks_rst/projection.rst',
                           'docs/_notebooks_rst/geoarray.rst',
                           ])

try:
    import matplotlib
except ImportError:
    # netCDF4 module is not present
    warnings.warn("matplotlib cannot be imported, skip plotting-related tests and the full documentation")
    collect_ignore.extend(['dimarray/plotting.py'])
    collect_ignore.extend(['docs/'])

# check for cartopy
try:
    import cartopy
except ImportError:
    # netCDF4 module is not present
    warnings.warn("cartopy cannot be imported, skip all coordinate transform tests related to cartopy")
    collect_ignore.extend(['dimarray/geo/crs.py','dimarray/geo/projection.py',
                           'dimarray/geo/tests/test_crs.py','dimarray/compat/cartopy.py',
                           'docs/_notebooks_rst/projection.rst'])

try:
    import iris.util
except ImportError:
    # netCDF4 module is not present
    warnings.warn("iris.util cannot be imported, skip all iris tests")
    collect_ignore.extend(['dimarray/convert/iris.py','dimarray/convert/tests/test_iris.py'])

try:
    import pandas
except ImportError:
    warnings.warn("pandas cannot be imported, skip tutorial and reshape doc (which involve pandas)")
    collect_ignore.extend(["docs/_notebooks_rst/tutorial.rst",'docs/_notebooks_rst/reshape.rst'])

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
