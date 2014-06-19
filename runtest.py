#!python
import sys, os
import pytest
import dimarray  # imported now so that no new version becomes imported

from tests.testing import main as run_doctests, testfile

# remove pyc files before testing: find . -name "*.pyc" -exec rm {} \;

# Uncomment one of the line below to change the limit of failed doctest results
# MyTestResults.maxfailed = 0
# MyTestResults.maxfailed = 1  # default
# MyTestResults.maxfailed = 10
# MyTestResults.maxfailed = 100

# first run unit tests using pytest
res = pytest.main()

# then check docstrings
test = run_doctests.main()

# test readme
test += testfile('README.rst')

test.summary()
