dimarray: array with labelled dimensions 
========================================

Check out and download the [tutorial as a notebook](http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb)

* Indexing
    * See also: `take`/`put`
* operations: alignment and broadcasting rules
* along-axis transformations
* missing values
* modify dimensions
    * numpy-like: `reshape`, `transpose`, `swapaxes`, `repeat`
    * new methods: `newaxis`, `broacast`, `group`/`ungroup`
    * functions: `align_axes`, `align_dims`, `broadcast_arrays`, `concatenate`, `aggregate`
* `Dataset` class: an ordered Dictionary of aligned aligned arrays
* re-indexing and interpolation
    * `reindex_axis`, `interp1d`
* I/O: netCDF
* from/to pandas

>>> a.to_pandas()   # doctest: +SKIP
time       1950      1951      1952      1953      1954
items                                                  
a      1.764052  0.400157  0.978738  2.240893  1.867558
b     -0.977278  0.950088 -0.151357 -0.103219  0.410599
c      0.144044  1.454274  0.761038  0.121675  0.443863


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

* DimArray			: main data structure (see alias `array`)
* Dataset		    	: ordered dictionary of DimArray objects
* read_nc, write_nc, summary_nc : netCDF I/O (DimArray and Dataset methods)
* Axis, Axes, GroupedAxis   : axis and indexing (under the hood)

And for things pandas does better (low-dimensional data analysis, `groupby`, 
I/O formats, etc...), just export via to_pandas() method (up to 4-D) (only
if pandas is installed of course - otherwise dimarray does not rely on pandas)
