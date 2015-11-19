Release notes draft for 0.2.0
=============================
- deprecated / removed:
    - `aggregate` (not useful)
    - `DimArray.groupby` (not really a multi-dimensional operation, alternative: dima.to_pandas().groupby(...) )
- modified behavior
    - multi-dimension indexing : now "orthogonal" by default. Use `DimArray.take(..., broadcast=True)` to broadcast arrays on the same axis, consistently with numpy.
- renamed:
    - `DimArray.group` => `DimArray.flatten`
    - `DimArray.ungroup` => `DimArray.unflatten`
    - `GroupedAxis` => `MultiAxis`
    - `align_axes` => `align`
- new features:
    - `align` : new `join=` parameter "inner" or "outer"
    - slicing on ordered numerical axis : a[2.:4.] on [1.5, 2.5, 3.5, 4.5] now works and return values on [2.5, 3.5]
    - `open_nc` to return a `DatasetOnDisk` : allow examination of datasets without actually reading it (somewhere between read_nc and the original netCDF4 module)
    - `DimArray.iloc`, `DimArray.loc`, `DimArray.sel`, `DimArray.isel`: convergence toward xray/pandas API
    - `DimArray.nloc` : like DimArray.loc but nearest-neighbor indexing
    - numpy.datetime64 can now be written to netCDF, and converted back (conversion adapted from on `xray.conventions` functions)
- internal:
    - indexing rewritten from the bottom up (basically: numpy.argsort + searchsorted and subsequent check, when more than one element is searched, `nonzero` otherwise)
        ==> much shorter source code 
        ==> more efficient calculations (reindexing-speed comparable with xray, a factor 2 slower than pandas)
        ==> no dependency on pandas anymore
    - netCDF I/O rewritten, too, now relies on DatasetOnDisk, DimArrayOnDisk, AxisOnDisk, AxesOnDisk, AttrsOnDisk classes.
    - AbstractDimArray, AbstractDataset and AbstractAxis classes from which DimArray and DimArrayOnDisk derive
        ==> indexing and formatting unified
- experimental:
    - dimarray.geo.crs.get_crs() : returns an instance of a cartopy CRS class based on a dictionary of CF conventions
    - dimarray.geo.transform() : transform and interpolate a dimarray from one coordinate system to another
    - dimarray.geo.transform_array() : transform and interpolate two dimarrays 

Temporary roadmap for future release(s)
---------------------------------------
Mostly, simplify and thin down the source code, make it more readable:
- dataset : do not make it an instance of OrderedDict with management of axes copy/refs as it is now, but make it a new object.
- simplify indexing in abstract classes (keep only orthogonal indexing, provide array index broadcasting as a separate function)
- simplify axes definition (allow only a few possible methods, to reduce the amount of checks)
- deprecate the dimarray.geo.GeoArray object, and flatten the sub-package for the user 
  (everything accessible from dimarray.geo (not crs etc...))
- review the library of functions provided, only keep the most useful ones
- simplify the cascade of classes in the netCDF I/O part
- remove the global option file?
- improve unit tests to have less redundancy, more readability
