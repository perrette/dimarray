"""
"""
import numpy as np
import dimarray as da
from dimarray import Dataset

def test():
    """ Test Dataset functionality

    >>> data = test() 
    >>> data['test2'] = da.DimArray([0,3],('source',['greenland','antarctica'])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: axes values do not match, align data first.                            
    Dataset: source(1)=greenland:greenland, 
    Got: source(2)=greenland:antarctica
    >>> data['ts']
    dimarray: 5 non-null elements (5 null)
    dimensions: 'time'
    0 / time (10): 1950 to 1959
    array([  0.,   1.,   2.,   3.,   4.,  nan,  nan,  nan,  nan,  nan])
    >>> data.to_array(axis='items')
    dimarray: 12250 non-null elements (1750 null)
    dimensions: 'items', 'lon', 'lat', 'time', 'source'
    0 / items (4): mymap to test
    1 / lon (50): -180.0 to 180.0
    2 / lat (7): -90.0 to 90.0
    3 / time (10): 1950 to 1959
    4 / source (1): greenland to greenland
    array(...)
    """
    import dimarray as da
    axes = da.Axes.from_tuples(('time',[1, 2, 3]))
    ds = da.Dataset()
    a = da.DimArray([[0, 1],[2, 3]], dims=('time','items'))
    ds['yo'] = a.reindex_like(axes)

    np.random.seed(0)
    mymap = da.DimArray.from_kw(np.random.randn(50,7), lon=np.linspace(-180,180,50), lat=np.linspace(-90,90,7))
    ts = da.DimArray(np.arange(5), ('time',np.arange(1950,1955)))
    ts2 = da.DimArray(np.arange(10), ('time',np.arange(1950,1960)))

    # Define a Dataset made of several variables
    data = da.Dataset({'ts':ts, 'ts2':ts2, 'mymap':mymap})
    #data = da.Dataset([ts, ts2, mymap], keys=['ts','ts2','mymap'])

    assert np.all(data['ts'].time == data['ts2'].time),"Dataset: pb data alignment" 

    data['test'] = da.DimArray([0],('source',['greenland']))  # new axis
    #data

    return data
