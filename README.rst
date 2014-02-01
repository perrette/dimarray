============================================================================
dimarray: array with labelled dimensions and axes, metadata and NaN handling
============================================================================

Forget the underlying array shape and concentrate on what you aim at doing !
The best of two worlds: numpy arrays and pandas DataFrame.

Inspired by (but does not rely on) pandas:

    * behave like a numpy array (operations, transformations)
    * labelled axes, NaN handling
    * automatic axis aligment for +-/* between two DimArray objects
    * similar api (`values`, `axes`,`reindex_axis`) 
    * group/ungroup methods to flatten any subset of dimensions into a 
      GroupedAxis object, in some ways similar to pandas' MultiIndex.
    * groupby method (planned)

But generalized to any dimension and augmented with new features:

    * intuitive multi-dimensional slicing/reshaping/transforms by axis name
    * arithmetics between arrays of different dimensions (broadcasting)
    * can assign weights to each axis (such as based on axis spacing)
      ==> `mean`, `var`, `std` can be weighted
    * in combination to `group`, can achieve area- or volumne- weighting
    * natural netCDF I/O  via netCDF4 python module (requires HDF5, netCDF4)
    * stick to numpy's api when possible (but with enhanced capabilities):
      `reshape`, `repeat`, `transpose`, `newaxis`, `squeeze`
      

Yet simpler with more control left to user (most attributes are non-protected, 
e.g. `values` can be overwritten), and organized around a small number of 
classes and methods:

    * DimArray			: main data structure (see alias `array`)
    * Dataset		    	: ordered dictionary of DimArray objects
    * read_nc, write_nc, summary_nc: netCDF I/O (DimArray and Dataset methods)
    * Axis, Axes, GroupedAxis   : axis and indexing (under the hood)

And for things pandas does better (low-dimensional data analysis, `groupby`, 
I/O formats, etc...), just export via to_pandas() method (up to 4-D) (only
if pandas is installed of course - otherwise dimarray does not rely on pandas)


Check out and download the notebook: 
http://nbviewer.ipython.org/github/perrette/dimarray/blob/master/dimarray.ipynb

Get started
-----------

>>> import numpy as np
>>> from dimarray import DimArray
>>> import dimarray as da

Define some dummy data representing 3 items "a", "b" and "c" over 5 years:

>>> np.random.seed(0) # just for the reproductivity of the examples below
>>> values = np.random.randn(3,5)

Defining a DimArray from there is pretty straightforward

>>> a = DimArray(values)   # fully automatic labelling
>>> a = DimArray(values, dims=['items', 'time'])  # axis values assumed [0, 1, 2,...]
>>> a = DimArray(values, axes=[list("abc"), np.arange(1950,1955)]) # "x0","x1" 
>>> a = DimArray(values, [('items',list("abc")), ('time',np.arange(1950,1955))])  # all labels
>>> a    
dimarray: 15 non-null elements (0 null)
dimensions: 'items', 'time'
0 / items (3): a to c
1 / time (5): 1950 to 1954
array([[ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799],
       [-0.97727788,  0.95008842, -0.15135721, -0.10321885,  0.4105985 ],
       [ 0.14404357,  1.45427351,  0.76103773,  0.12167502,  0.44386323]])

For convenience, it is also possible to define an array via keyword arguments
(but note this is unambiguous only when axes have different size, see tutorial)

>>> b = DimArray.from_kw(values, items=list("abc"), time=np.arange(1950,1955))
>>> c = da.array(values, items=list("abc"), time=np.arange(1950,1955)) # alias
>>> np.all(a == b)
True

Can access underlying numpy array via `values` attributes

>>> a.values  # doctest: +ELLIPSIS
array(...)

And the axes, too:

>>> a.time
array([1950, 1951, 1952, 1953, 1954])
>>> a.items
array(['a', 'b', 'c'], dtype=object)
>>> np.all( a.time == a.axes["time"].values )
True

Easy slicing (take)

>>> a.take(1952, axis='time')  #  classical numpy-like form
dimarray: 3 non-null elements (0 null)
dimensions: 'items'
0 / items (3): a to c
array([ 0.97873798, -0.15135721,  0.76103773])
>>> a.take({'time':1952, 'items':"b"})  # dict-like
-0.15135720829769789
>>> a.take({'time':slice(1952,1954)})
dimarray: 9 non-null elements (0 null)
dimensions: 'items', 'time'
0 / items (3): a to c
1 / time (3): 1952 to 1954
array([[ 0.97873798,  2.2408932 ,  1.86755799],
       [-0.15135721, -0.10321885,  0.4105985 ],
       [ 0.76103773,  0.12167502,  0.44386323]])
>>> np.all(a[:, 1952:1954] == a.take(slice(1952,1954),axis='time'))
True
>>> a.ix[:, 2:5] # integer access
dimarray: 9 non-null elements (0 null)
dimensions: 'items', 'time'
0 / items (3): a to c
1 / time (3): 1952 to 1954
array([[ 0.97873798,  2.2408932 ,  1.86755799],
       [-0.15135721, -0.10321885,  0.4105985 ],
       [ 0.76103773,  0.12167502,  0.44386323]])


Arithmetics with reshaping and axis alignment

>>> ts = da.array(np.random.randn(5), time=np.arange(1950, 1955))
>>> a * ts   # not commutative !  # doctest: +ELLIPSIS
dimarray: 15 non-null elements (0 null)
dimensions: 'items', 'time'
0 / items (3): a to c
1 / time (5): 1950 to 1954
array(...)
>>> mymap = da.array(np.random.randn(5,10), lon=np.linspace(0,360,10), lat=np.linspace(-90,90,5))
>>> mycube = mymap * ts   # doctest: +ELLIPSIS
>>> mycube
dimarray: 250 non-null elements (0 null)
dimensions: 'lat', 'lon', 'time'
0 / lat (5): -90.0 to 90.0
1 / lon (10): 0.0 to 360.0
2 / time (5): 1950 to 1954
array(...)

All numpy transforms work (with NaN checking)

>>> mycube.mean(axis="time") 
dimarray: 50 non-null elements (0 null)
dimensions: 'lat', 'lon'
0 / lat (5): -90.0 to 90.0
1 / lon (10): 0.0 to 360.0
array([[-0.55224596,  0.14138647,  0.18698915, -0.16054025,  0.49097838,
	-0.31459881,  0.00989818, -0.04049038,  0.33156071,  0.31784202],
       [ 0.03351721,  0.08180163, -0.19203997, -0.42847286, -0.07525807,
	 0.03382038,  0.26612838,  0.2600909 , -0.08378399, -0.06539214],
       [-0.22681608, -0.30716894, -0.36908914,  0.4219789 , -0.11024461,
	-0.09476135, -0.27099645,  0.1681816 , -0.34910776, -0.04601858],
       [-0.19370143,  0.0836922 , -0.11049401, -0.25538659, -0.00609619,
	 0.09265393,  0.01438857,  0.06542873, -0.13721238, -0.07846578],
       [-0.14546222, -0.07777617, -0.17589445, -0.37341809,  0.03837966,
	-0.08691061, -0.35263378,  0.10010601, -0.19626081,  0.01123649]])
>>> a.values.mean(axis=1) == a.mean(axis="time").values  
array([ True,  True,  True], dtype=bool)

Can also provide a subset of several dimensions as argument to operate on flattened array.

>>> mycube.mean(axis=("lat","lon"))
dimarray: 5 non-null elements (0 null)
dimensions: 'time'
0 / time (5): 1950 to 1954
array([-0.08412077, -0.37666392,  0.0517213 , -0.07892575,  0.2153213 ])


NetCDF I/O

>>> a.write("test.nc","myvar", mode='w') # write to netCDF4
write to test.nc
>>> da.summary_nc("test.nc") # check the content
test.nc:
-------
Dataset of 1 variable
dimensions: u'items', u'time'
0 / items (3): a to c
1 / time (5): 1950 to 1954
myvar : items, time
>>> dataset = da.read_nc("test.nc") # read in a Dataset class
read from test.nc
>>> dataset
Dataset of 1 variable
dimensions: u'items', u'time'
0 / items (3): a to c
1 / time (5): 1950 to 1954
myvar: items, time
>>> np.all(dataset["myvar"] == a)
True

Easy interfacing with pandas

>>> a.to_pandas()
time       1950      1951      1952      1953      1954
items                                                  
a      1.764052  0.400157  0.978738  2.240893  1.867558
b     -0.977278  0.950088 -0.151357 -0.103219  0.410599
c      0.144044  1.454274  0.761038  0.121675  0.443863

E.g. for plotting:

a.to_pandas().plot()  

More complete documentation will follow...