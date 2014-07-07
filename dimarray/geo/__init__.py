""" This module is experimental, and will likely undergo many changes
"""
# import all basic DimArray-like objects
# so that one can also do: import dimarray.geo as da
from dimarray import *

# Add GeoArray to the list
from geoarray import GeoArray

# Overwrite with specific GeoArray functions
from ncio import *
