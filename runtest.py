#!/usr/bin/python
import pytest
import dimarray.tests as _doctests

# remove pyc files before testing: 'find . -name "*.pyc" -exec rm {} \;'

# first run unit tests using pytest
res = pytest.main()

# then check docstrings
_doctests.main()
