dimarray documentation
======================

.. dimarray documentation master file, created by
   sphinx-quickstart on Wed Jun 18 01:41:33 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

.. A `numpy` array with labelled axes and dimensions, in the spirit of 
.. `pandas` but not restricted to low-dimensional arrays, with improved
.. indexing, and using the full power of having dimension names 
.. (see below "Comparisons with other packages").
.. 
.. Having axis name and axis values allow on-the-fly axis alignment and 
.. dimension broadcasting in basic operations (addition, etc...), 
.. so that rules can be defined for nearly every sequence of operands. 
.. 
.. A natural I/O format for such an array is netCDF, common in geophysics, which rely on 
.. the netCDF4 package. Other formats are under development (HDF5). Metadata are also 
.. supported, and conserved via slicing and along-axis transformations.

.. toctree::
   :maxdepth: 2

   introduction.rst
   _build_rst/getting_started.rst
   _build_rst/examples.rst
   reference.rst
   reference_api.rst
   comparison_table.rst
   other_packages.rst

Further development
===================
All suggestions for improvement very welcome, please file an `issue` on github:
https://github.com/perrette/dimarray/ for further discussion.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

