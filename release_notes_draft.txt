Release notes draft for 0.2
---------------------------
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
