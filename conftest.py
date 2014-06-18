import dimarray as da

collect_ignore = ["build", "tests/test_mpl.py"]

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

