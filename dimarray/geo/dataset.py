from dimarray.geo import GeoArray
import dimarray.dataset as ds

class Dataset(ds.Dataset):
    """ Subclass of dimarray's Dataset that stores GeoArray instances
    """
    _constructor = GeoArray
