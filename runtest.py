#!python
import sys, os
import pytest
import dimarray.tests as _doctests
from dimarray.testing import testmod, MyTestResults

# remove pyc files before testing: 'find . -name "*.pyc" -exec rm {} \;'

# Uncomment one of the line below to change the limit of failed doctest results
# MyTestResults.maxfailed = 0
# MyTestResults.maxfailed = 1  # default
# MyTestResults.maxfailed = 10
# MyTestResults.maxfailed = 100

# first run unit tests using pytest
res = pytest.main()

# then check docstrings
test = _doctests.main()

# and the old docstrings
sys.path.append('tests')
import olddocs 
test += testmod(olddocs)

# report test results
print "============================="
print "Failed   : {}\nAttempted: {}".format(test.failed, test.attempted)
print "============================="
if test.failed > 0:
    print ">>>>>>>>> Failed Tests"
