Ref Table numpy vs dimarray
===========================

Table of correspondence between numpy ndarray and dimarray's functions and methods

==================  ==================  ================    
array creation
--------------------------------------------------------
numpy               dimarray            comments
==================  ==================  ================    
array               DimArray            In dimarray need to provide axes information in addition to values.
-                   DimArray.from_kw    Same as DimArray() but provide axes as key-words.
-                   array               same as DimArray()
-                   array_kw            same as DimArray.from_kw()
zeros               zeros               These functions are similar to numpy except that they require `axes` parameter, or a shape= parameter for automatic labelling.
ones                ones                  
empty               empty                 
zeros_like          zeros_like            
ones_like           ones_like             
empty_like          empty_like            
==================  ==================  ================    

.. note:: The `array` and `array_kw` forms (to be used as da.array()) are attempts to make the array definition less verbose. They are experimental and may change in the future.

==================  ==================  ================    
reshaping
--------------------------------------------------------
numpy               dimarray            comments
==================  ==================  ================    
a.T                 a.T                 Transpose a 2-dimensional array
a.transpose()       a.transpose()       Transpose or permute array. In dimarray also accept axis names 
a.swapaxes()        a.swapaxes()        Swap two axes. In dimarray also accept axis names. e.g. a.swapaxes('time', 0) to bring the 'time' dimension as first axis to ease indexing.
a.reshape()         a.reshape()         Change array shape without changing the size. There are a few differences in dimarray compared to numpy:
                                        - a dimension cannot be broken down (e.g. 4 => 2x2)
                                        - the full shape of the array is given via axis names
                                        e.g. a.reshape('time,percentile','scenario') will flatten (`group`) the dimensions `time` and `percentile`
                                        to end up with a 2-D array, and transpose the array as necessary to get to the desired shape.
                                        If only transposing (permutation) is needed, the use of `transpose` is preferred for clarity.
    -               a.group()           Flatten two axes into one: it is for `reshape` what `swapaxes` is to `transpose`.
    -               a.ungroup()         Inflate two or more "grouped" axes (undo a.group()). 
a.flatten()         a.flatten()         Flatten array. In dimarray the axes are transformed into tuples (`GroupedAxis`). 
a[np.newaxis]       a.newaxis()         In numpy, add a singleton dimension, useful for broadcasting 
                                        in an operation. In dimarray, broadcasting is based on dimension 
                                        names and therefore streamlined without the need to profide this 
                                        extra-information, make this option less relevant in the public API. 
                                        In dimarray this is a method since it requires the name of the new axis,
                                        and by extension, if the new axis' values are also provided it can also 
                                        combines functionality of `repeat`. 
a.squeeze()         a.squeeze()         idem, but also accept axis names (opposite of `newaxis`)
a.repeat()          a.repeat()          In dimarray it's mostly an internal method that only
                                        works on singleton dimensions. This is of no 
                                        much practical use. Use `newaxis` instead.
broadcast()         a.broadcast()       Dimarray's method similar to numpy's function. Add or remove singleton axes to make it match another array's 
                                        dimensions, but **without repeating**
                                        (so that the shapes do not necessarily match, but it is ready for binary operations in a numpy sense)
                                        In dimarray, the broadcast method can also transpose axes to match dimension ordering.
broadcast_arrays()  broadcast_arrays()  Functions. Like the above, but also repeat the arrays if necessary to match the shape.
==================  ==================  ================    

.. note:: The names `group` and `ungroup` may be confusing and could change in the future (e.g. to flatten and inflate, or unflatten)

The methods below are mostly similar across the packages, but dimarray also accepts axis name instead of axis rank as `axis=`. 
An optional `skipna=` parameter can be provided to ignore nans (default to `False`). Note also that in many cases, 
when a `tuple` of axis names is provided the array is first partially flattened (grouped axis) before the dimension is reduced

==================  ==================  ================    
reduce, accumulate (along-axis transformation)
--------------------------------------------------------
numpy               dimarray            comments
==================  ==================  ================    
a.max()             a.max()             
a.min()             a.min()             
a.ptp()             a.ptp()             
a.median()          a.median()          
a.all()             a.all()             
a.any()             a.any()             
a.prod()            a.prod()            
a.sum()             a.sum()             
a.mean()            a.mean()            with optional "weights=" parameter to transform into a weighted mean, which is also checked from Axis attribute.
a.std()             a.std()             idem
a.var()             a.var()             idem                         
a.argmax()          a.argmax()          in dimarray, returns axis value of max instead of integer position on the axis
a.argmin()          a.argmin()          idem
a.cumsum()          a.cumsum()          
a.cumprod()         a.cumprod()         
diff(a,...)         a.diff()            as method, and with `scheme=` parameter ("forward", "centered", "backward")

==================  ==================  ================    
