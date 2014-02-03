dimarray: array with labelled dimensions 
========================================

Download dimarray on `github <https://github.com/perrette/dimarray/>`_
or just take a look at this notebook on
`nbviewer <http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb>`_

Table of Content
~~~~~~~~~~~~~~~~

-  `Get started <#Get-started>`_
-  `Alternative definitions <#Alternative-definitions>`_
-  `Indexing <#Indexing>`_

   -  `Basics: integer, array, slice <#Basics:-integer,-array,-slice>`_
   -  `Modify array values <#Modify-array-values>`_
   -  `take and put methods <#take-and-put-methods>`_

-  `Numpy Transformations <#Numpy-Transformations>`_
-  `Missing values <#Missing-values>`_
-  `Modify array shape <#Modify-array-shape>`_

   -  `transpose <#transpose>`_
   -  `newaxis <#newaxis>`_
   -  `reshape <#reshape>`_
   -  `group and ungroup
      [Experimental] <#group-and-ungroup-[Experimental]>`_

-  `Repeat and broadcast: align
   dimensions <#Repeat-and-broadcast:-align-dimensions>`_

   -  `repeat <#repeat>`_
   -  `broadcast <#broadcast>`_
   -  `broadcast\_arrays <#broadcast_arrays>`_

-  `Reindexing: align axes <#Reindexing:-align-axes>`_

   -  `reindex\_axis <#reindex_axis>`_
   -  `reindex\_like <#reindex_like>`_
   -  `Interpolation <#Interpolation>`_
   -  `align\_axes <#align_axes>`_

-  `Join arrays <#Join-arrays>`_

   -  `concatenate arrays along existing
      axis <#concatenate-arrays-along-existing-axis>`_
   -  `join arrays along new axis <#join-arrays-along-new-axis>`_
   -  `aggregate arrays of varying dimensions
      [Experimental] <#aggregate-arrays-of-varying-dimensions-[Experimental]>`_

-  `Operations <#Operations>`_

   -  `Basic Operations <#Basic-Operations-------->`_
   -  `Operation with data alignment <#Operation-with-data-alignment->`_

-  `Dataset <#Dataset>`_
-  `NetCDF I/O <#NetCDF-I/O>`_
-  `Experimental Features <#Experimental-Features>`_

   -  `Metadata <#Metadata>`_
   -  `Weighted mean <#Weighted-mean>`_
   -  `Compatibility with pandas and
      larry <#Compatibility-with-pandas-and-larry>`_

-  `doctest framework <#doctest-framework>`_


Notebook:
---------
http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb


Summary:
--------

Inspired by (but does not rely on) pandas:

* behave like a numpy array (operations, transformations)
* labelled axes, NaN handling
* automatic axis aligment for +-/* between two DimArray objects
* similar api (`values`, `axes`,`reindex_axis`) 
* group/ungroup methods to flatten any subset of dimensions into a 
  GroupedAxis object, in some ways similar to pandas' MultiIndex.

But generalized to any dimension and augmented with new features:

* intuitive multi-dimensional slicing/reshaping/transforms by axis name
* arithmetics between arrays of different dimensions (broadcasting)
* can assign weights to each axis (such as based on axis spacing)
  ==> `mean`, `var`, `std` can be weighted
* in combination to `group`, can achieve area- or volumne- weighting
* natural netCDF I/O  via netCDF4 python module (requires HDF5, netCDF4)
* stick to numpy's api when possible (but with enhanced capabilities):
  `reshape`, `repeat`, `transpose`, `newaxis`, `squeeze`
      

Organized around a small number of classes and methods:

* DimArray			: main data structure 
* Dataset		    	: ordered dictionary of DimArray objects
* read_nc, write_nc, summary_nc : netCDF I/O (DimArray and Dataset methods)
* Axis, Axes, GroupedAxis   : axis and indexing (under the hood)

And for things pandas does better (low-dimensional data analysis, `groupby`, 
I/O formats, etc...), just export via to_pandas() method (up to 4-D) (only
if pandas is installed of course - otherwise dimarray does not rely on pandas)
