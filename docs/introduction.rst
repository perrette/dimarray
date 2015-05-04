.. include:: ../README.rst

The ecosystem of labelled arrays
--------------------------------

A brief overview of the various array packages around. The listing is chronological.
dimarray is strongly inspired from pandas and larry packages.

`numpy`_ provides the basic array object, transformations and so on. It does
not include axis labels and has limited support for missing values (NaNs). 
An extension, numpy.ma, adds a `mask` attributes and skip NaNs in 
transformations. 

`larry`_ was pioneer as labelled array, it skips nans in transforms
and comes with a wealth of built-in methods. It is very computationally-efficient
via the use of bottleneck. For now it does not support naming dimensions.

`pandas`_ is an excellent package for low-dimensional data analysis, 
supporting many I/O formats and axis alignment (or "reindexing") 
in binary operations. It is mostly limited to 2 dimensions (DataFrame), 
or up to 4 dimensions (Panel, Panel4D).  

`iris`_ looks like a very powerful package to manipulate geospatial data with 
metadata, netCDF I/O, performing grid transforms etc..., but it is quite a jump 
from numpy's `ndarray` in term of syntax and requires a bit of learning. 

`dimarray`_, like iris, considers dimension names as a fundamental property
of an array, and as such supports netCDF I/O format. It makes use of it 
in binary operations (broadcasting), transforms and indexing. 
It includes some of the nice features of pandas (e.g. axis alignment, optional
nan skipping) but extends them to N dimensions, with a behaviour closer 
to a numpy array. Some geo features are planned (weighted mean for latitude, 
indexing modulo 360 for longitude, basic regridding) but dimarray should remain broad in scope.

`xray`_ has many similarities with `dimarray`, but has a stronger focus 
on following the `Common Data Model`_, i.e. its primary object is the Dataset
whereas dimarray is centered on the multidimensional array, with the Dataset 
being more of an extension (a dictionary of DimArrays).
Moreover, `xray` is tightly integrated with `pandas` (it uses pandas Index 
as coordinates), whereas `dimarray` simply provides methods to exchange
data from and to `pandas` objects, but does not actually rely on pandas.
This makes dimarray+dependencies somewhat more lightweight than xray.
In term of speed (indexing...), things are comparable - since all
computationally-intensive operations in `dimarray` rely on fast numpy.
`xray` is a very thorough project (e.g. unit tests etc...) with a commited 
author, so if your primary focus is to work with netCDF, access openDAP 
data and so on, you may want to go that way. Try it out!

`spacegrids`_ is a promising new package with focus on geospatial grids. 
It intends to streamline a number of operations such as
derivations, integration, regridding by proposing an algebra on 
between arrays and axes (grids). It also includes a project 
management utility for netCDF files. 

.. _numpy: http://docs.scipy.org/doc/numpy/user
.. _larry: http://berkeleyanalytics.com/la
.. _pandas: http://pandas.pydata.org 
.. _iris: http://scitools.org.uk/iris
.. _dimarray: dimarray.readthedocs.org
.. _xray: xray.readthedocs.org
.. _Common Data Model: http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM
.. _spacegrids: https://github.com/willo12/spacegrids

.. seealso:: :meth:`DimArray.to_pandas() <dimarray.DimArray.to_pandas>` and :meth:`DimArray.from_pandas() <dimarray.DimArray.from_pandas>`.
