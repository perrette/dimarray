""" Subclass of DimArray specialized for geo-applications

==> dimensions always remain sorted: ('items','time','lat','lon','height','sample')
==> subset of methods with specific plotting behaviour: e.g. TimeSeries, Map
"""
from dimarray import DimArray

class GeoArray(DimArray):
    _order = ('items','time','lat','lon','height','sample')
    _dims = None

    def __init__(self, values=None, *axes, **kwargs):
	"""
	"""
	assert (len(axes) == 0 or len(kwargs) ==0), "cant provide axes both as kwargs and list"
	assert self._dims is None or (len(axes) == self._dims or len(kwargs) == len(self._dims)), "dimension mismatch"
	if len(kwargs) > 0:
	    for k in kwargs:
		if k not in self._order:
		    raise ValueError("unknown dimension, please provide as axes")
	    if self._dims is not None:
		axes = [k,kwargs[k] for k in self._dims if k in kwargs]
	    else:
		axes = [k,kwargs[k] for k in self._order if k in kwargs]

	else:
	    if self._dims is not None:
		assert tuple(ax.name for ax in axes) == self._dims, "dimension mismtach"

	super(GeoArray, self).__init__(values, axes)
	for k in kwargs: self.setncattr(k, kwargs[k])
    
    @classmethod
    def _constructor(cls, values, axes, **kwargs):
	dims = tuple(ax.name for ax in axes)
	class_ = _get_geoarray_cls(dims)
	if class_ is not None:
	    obj = class_(values, *axes)
	else:
	    obj = cls(values, *axes)
	for k in kwargs: obj.setncattr(k, kwargs[k])
	return obj

def _get_geoarray_cls(dims, globs=None):
    """ look whether a particular pre-defined array matches the dimensions
    """
    if globs is None: globs = globals()
    cls = None
    for obj in globs.keys():
	if isinstance(obj, globals()['GeoArray']):
	    if tuple(dims) == cls._dimensions:
		cls = obj

    return cls


class TimeSeries(GeoArray):
    _dims = ('time',)

class Map(GeoArray):
    _dims = ('lat','lon')

class TimeMap(GeoArray):
    _dims = ('time','lat','lon')

class Sample(GeoArray):
    _dims = ('sample',)
