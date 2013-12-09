""" Definition of dimensions and a few classical classes

Note one could be tempted to add Series and DataFrame like in pandas
but this would be contrary to the spirit of a Dimarray, as 
`columns` and `index` are not semantic labels (and therefore 
they are not conserved in pandas)
"""
#
# Add a few predefined class (e.g. Map, TimeSeries etc...)
# see the definition module 
#
class Defarray(Dimarray):
    """ Array with pre-defined dimensions
    
    This class makes initializing a dimarray easier
    """
    _dimensions = None

    def __init__(self, values, *axes, **kwargs): 
	""" init
	"""
	assert self._dimensions is not None, "Need subclassing !"
	if len(axes) == 0: axes = None
	super(Defarray, self).__init__(values, [axes, self._dimensions], **kwargs)

	# check dimensions
	for d1, d2 in zip(self._dimensions, self.axes.names):
	    if d1 != d2: 
		print "required: ",self._dimensions
		print "provided: ",self.axes.names
		raise Exception("dimension mismatch !!")

#
# A few example subclasses from the geophysics
#
class TimeSeries(Defarray):
    _dimensions = ("time",)

class Map(Defarray):
    """ For map it makes no sense to plot lines like for a DataFrame
    we therefore overload the plot method
    """
    _dimensions = ("lat","lon")
    #plot = Defarray.contourf
