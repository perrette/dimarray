.. include:: ../README.rst

dimarray's niche in the custom arrays ecosystem
-----------------------------------------------

dimarray is strongly inspired from previous work, especially from pandas and larry packages.
This section attempts to highlight some of the conceptual differences between them. 

pandas 
^^^^^^
pandas is an excellent package for low-dimensional data analysis, 
with many I/O features, but is mostly limited to 2 dimensions (DataFrame), 
or up to 4 dimensions (Panel, Panel4D). 

`dimarray` includes
some of the nice `pandas` features, such as indexing on axis values, 
automatic axis alignment, intuitive string representation,
or a parameter to ignore nans in axis reduction operations. 
`dimarray` extends these functionalities to any number 
of dimensions. 

In general, `dimarray` is designed to be more consistent with 
`numpy`'s ndarray, whereas `pandas` is somewhat between a dictionary and 
a numpy array. One consequence is that standard indexing with `[]` can be 
multi-dimensional, another is that iteration (`__iter__` method) 
is on sub-arrays and not on axis values (the keys). 

`dimarray` comes with `to_pandas` and `from_pandas`
methods to use the most of each of the packages (also supports `MultiIndex`
via the equivalent `GroupedAxis` object). For convenience, a `plot`
method is defined in `dimarray` as an alias for to_pandas().plot().

larry
^^^^^
larry was pioneer as labelled array, it skips nans in most transforms
and comes with a wealth of built-in methods. It is very computationally-efficient
via the use of bottleneck. It is a bit less intuitive than `dimarray` or `pandas` 
as far as indexing is concerned - but it is a matter of taste, 
and does not support naming dimensions.
From the structure (array-like), dimarray is closer to larry than to pandas.
        

iris
^^^^
iris looks like a very powerful package to manipulate geospatial data with 
metadata, netCDF I/O, performing grid transforms etc..., but it is quite a jump 
from numpy's `ndarray` and requires a bit of learning. 

In contrast, `dimarray` is more general and intuitive for python users. `dimarray`
also comes with netCDF I/O capability and may gain a few geospatial features 
(weighted mean for lon/lat, 360 modulo for lon, regridding, etc...) as a subpackage 
**dimarray.geo** -- and why not an interface to `iris`.

conclusion
^^^^^^^^^^
Having a focus on dimension names and axis values instead of axis rank and position 
of elements along an axis is a strong feature of `dimarray`. This determines
fundamental behaviours such as automatic matching of dimensions during a binary 
operation, but also brings about convenience like passing axis name instead of axis
rank in many of the methods (`take`, `put`, `reshape`, `sum`, ...).
This has proven useful to write pretty generic code with arrays of various shape 
which all share a few dimensions `time`, `lon`, `lat`, `model`, `scenario`, `sample`, 
`percentile` and so on - as long as one sticks to these names in the particular bit
of code of course.
