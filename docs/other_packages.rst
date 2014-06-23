Comments on other packages
--------------------------

dimarray is strongly inspired from previous work, especially from pandas and larry packages.
This section attempts to highligh some of the conceptual differences between them, some of 
which are not likely to vanish with the future releases. Please correct if I'm wrong here.

- **pandas**
    ...is an excellent package for low-dimensional data analysis, 
    with many I/O features, but is mostly limited to 2 dimensions (DataFrame), 
    or up to 4 dimensions (Panel, Panel4D). `dimarray` includes
    some of the nice `pandas` features, such as indexing on axis values, 
    automatic axis alignment, intuitive string representation,
    or a parameter to ignore nans in axis reduction operations. 
    `dimarray` extends these functionalities to any number 
    of dimensions. In general, `dimarray` is designed to be more consistent with 
    `numpy`'s ndarray, whereas `pandas` is somewhat between a dictionary and 
    a numpy array. One consequence is that standard indexing with `[]` can be 
    multi-dimensional, another is that iteration is on sub-arrays and not on 
    axis values (the keys). `dimarray` comes with `to_pandas` and `from_pandas`
    methods to use the most of each of the packages (also supports `MultiIndex`
    via the equivalent `GroupedAxis` object). For convenience, a `plot`
    method is defined in `dimarray` as an alias for to_pandas().plot().

- **larry** 
    ...was pioneer as labelled array, it skips nans in along-axis transforms
    and comes with a wealth of built-in methods. It is very computationally-efficient
    via the use of bottleneck. It is a bit less intuitive than `dimarray` or `pandas` 
    as far as indexing is concerned, and does not support naming dimensions.
    From the structure (array-like), `dimarray` is closer to larry than to pandas.
        

Compared with these two pacakges, `dimarray` adds the possibility of passing axis 
name to the various methods, instead of simply axis rank. Having a focus on dimension
names and axis values instead of axis rank and position of elements along an axis
is a strong feature of `dimarray`. This applies for
instance to along-axis operation, `take` and `put` methods, or reshaping operations.
Additionally, `dimarray` is to my knowledge the only package supporting automatic
dimension broadcasting for any two operands. This has proven useful to write pretty
generic code with arrays of various shape which all share a few dimensions `time`, 
`lon`, `lat`, `model`, `scenario`, `sample`, `percentile` and so on.

- **iris** 
    ...looks like a very powerful package to manipulate geospatial data with 
    metadata, netCDF I/O, performing grid transforms etc..., but it is quite a jump 
    from numpy's `ndarray` and requires a bit of learning. 
    In contrast, `dimarray` is more general and intuitive for python users. `dimarray`
    also comes with netCDF I/O capability and may gain a few geospatial features 
    (weighted mean for lon/lat, 360 modulo for lon, regridding, etc...) as a subpackage 
    **dimarray.geo** -- and why not an interface to `iris`.
