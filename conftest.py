import dimarray as da

collect_ignore = ["build", "tests/testing.py", "tests/test_mpl.py"]

try:
    import netCDF4
except ImportError:
    # netCDF4 module is not present
    collect_ignore.append('dimarray/io/nc.py')

#
# could not find a way to provide a globs= parameter to doctest in the py.test framework
# so the --doctest-glob option is kind of useless
# it is handled by testing.run_doctests
#
import tests.testing

# for inclusion of the following modules in docstring tes
tests.testing.MAXFAILED = 5  # number of failed example before quitting
tests.testing.EXCLUDE = ['dimarray.io.nc']  # module to exclude in recursive searching
tests.testing.INCLUDE = ['tests.test_core', 'tests.testing', 'tests.test_nc', 'tests.test_operations', 'tests.olddocs'] # additionaly test docstring on those

# test on python version before executing this file

#if sys.version_info[0] > 2:
#        collect_ignore.append("pkg/module_py2.py")

# nice dimarray display?
def pytest_assertrepr_compare(op, left, right):
    if (isinstance(left, da.DimArray) or isinstance(left, np.ndarray)) and (isinstance(right, da.DimArray) or isinstance(right.np.ndarray)):
    #and op == "==":

	res = ['Comparing {} and {} instances:'.format(type(left).__name__, type(right).__name__)] \
		+ [''] \
		+ repr(left).split('\n') \
		+ ['']  \
		+ [' :: does not compare with :: ']  \
		+ ['']  \
		+ repr(right).split('\n')

    return res

