""" Definition of dimensions and a few classical classes

Note one could be tempted to add Series and DataFrame like in pandas
but this would be contrary to the spirit of a Dimarray, as 
`columns` and `index` are not semantic labels (and therefore 
they are not conserved in pandas)
"""
from core import Dimarray, Axis, Axes
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


    @staticmethod
    def _constructor(values, axes, **metadata):
	""" Internal API for the constructor: check whether a pre-defined class exists

	values	: numpy-like array
	axes	: Axes instance 

	This static method is used whenever a new Dimarray needs to be instantiated
	for example after a transformation.

	This makes the sub-classing process easier since only this method needs to be 
	overloaded to make the sub-class a "closed" class.
	"""
	assert isinstance(axes, list), "Need to provide a list of Axis objects !"

	# scalar
	if len(axes) == 0:
	    return Dimarray(values, axes, **metadata)

	assert isinstance(axes[0], Axis), "Need to provide a list of Axis objects !"
	#assert isinstance(axes, Axes), "Need to provide an Axes object !"

	# loop over all variables defined in defarray
	cls = _get_defarray_cls([ax.name for ax in axes])

	# initialize with the specialized class
	if cls is not None:
	    new = cls(values, *axes, **metadata)

	# or just with the normal class constructor
	else:
	    new = Dimarray(values, axes, **metadata)
	    return new

#
# A FEW EXAMPLE SUBCLASSES FROM GEOPHYSICS
#
class TimeSeries(Defarray):
    _dimensions = ("time",)

class Map(Defarray):
    """ For map it makes no sense to plot lines like for a DataFrame
    we therefore overload the plot method
    """
    _dimensions = ("lat","lon")
    #plot = Defarray.contourf


def _get_defarray_cls(dims):
    """ look whether a particular pre-defined array matches the dimensions
    """
    import defarray
    cls = None
    for obj in vars(defarray): 
	if isinstance(obj, defarray.Defarray):
	    if tuple(dims) == cls._dimensions:
		cls = obj

    return cls
