""" Subclass of DimArray specialized for geo-applications

==> dimensions always remain sorted: ('items','time','lat','lon','height','sample')
==> subset of methods with specific plotting behaviour: e.g. TimeSeries, Map
"""
from dimarray import DimArray, Axes, Axis

def is_longitude(nm):
   return nm.lower() in ("lon","long", "longitude", "lons", "longitudes")

def is_latitude(nm):
   return nm.lower() in ("lat", "latitude", "lats", "latitudes")

class GeoArray(DimArray):
    """ array for geophysical application
    
    - automatically assign a weight to lon/lat
    - time, lat, lon order always maintained
    - can input time, lat, lon as keyword arguments
    """
    _order = ('time','lat','lon')

    def __init__(self, values=None, axes=None, time=None, lat=None, lon=None, **kwargs):
	""" 
	"""
	keyword = (time is not None or lat is not None or lon is not None)
	assert not (axes is not None and keyword), "can't input both `axes=` and keyword arguments!"

	# construct the axes
	if keyword:
	    axes = Axes()
	    for k in self._order:
		val = locals()[k]
		if val is None: continue
		axes.append(Axis(val, k))

	super(GeoArray, self).__init__(values, axes, **kwargs) # order dimensions

	# add weight on latitude
	for ax in self.axes:
	    if is_latitude(ax.name):
		ax.weight = lambda x: np.cos(np.radians(x))

#def _get_geoarray_cls(dims, globs=None):
#    """ look whether a particular pre-defined array matches the dimensions
#    """
#    if globs is None: globs = globals()
#    cls = None
#    for obj in globs.keys():
#	if isinstance(obj, globals()['GeoArray']):
#	    if tuple(dims) == cls._dimensions:
#		cls = obj
#
#    return cls


#class CommonGeoArray(GeoArray):
#    #pass
#    _order = ('items','time','lat','lon','height','sample')
#    def __init__(self, values=None, *axes, **kwargs):
#	"""
#	"""
#	assert (len(axes) == 0 or len(kwargs) ==0), "cant provide axes both as kwargs and list"
#	assert self._dims is None or (len(axes) == self._dims or len(kwargs) == len(self._dims)), "dimension mismatch"
#	if len(kwargs) > 0:
#	    for k in kwargs:
#		if k not in self._order:
#		    raise ValueError("unknown dimension, please provide as axes")
#	    if self._dims is not None:
#		axes = [k,kwargs[k] for k in self._dims if k in kwargs]
#	    else:
#		axes = [k,kwargs[k] for k in self._order if k in kwargs]
#
#	else:
#	    if self._dims is not None:
#		assert tuple(ax.name for ax in axes) == self._dims, "dimension mismtach"
#
#	super(CommonGeoArray, self).__init__(values, axes)
#	for k in kwargs: self.setncattr(k, kwargs[k])
#    
#    @classmethod
#    def _constructor(cls, values, axes, **kwargs):
#	dims = tuple(ax.name for ax in axes)
#	class_ = _get_geoarray_cls(dims)
#	if class_ is not None:
#	    obj = class_(values, *axes)
#	else:
#	    obj = cls(values, *axes)
#	for k in kwargs: obj.setncattr(k, kwargs[k])
#	return obj

#
#class TimeSeries(GeoArray):
#    _dims = ('time',)
#
#class Map(GeoArray):
#    _dims = ('lat','lon')
#
#class TimeMap(GeoArray):
#    _dims = ('time','lat','lon')
#
#class Sample(GeoArray):
#    _dims = ('sample',)
