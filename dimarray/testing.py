""" A few functions useful for testing
"""
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import dimarray as da
from dimarray.tools import anynan

SEED = None
#
# Create numpy arrays of various types
#
def create_array(shape, dtype, seed=SEED):
    """ create an array of desired type """
    np.random.seed(seed) # set random generator state
    values = np.random.rand(*shape)
    if dtype is bool:
        values = values > 0.5
    else:
        values = np.asarray(values, dtype=dtype)
    return values

def create_scalar(dtype, seed=SEED):
    return create_array((1,), dtype=t, seed=seed)[0]

def create_array_axis(size, dtype, regular=True, seed=SEED):
    """ create an axis of desired type """
    np.random.seed(seed) # set random generator state
    values = np.arange(size) + 10*np.random.rand() # random offset
    if not regular:
        np.random.seed(seed) # set random generator state
        values = np.random.shuffle(values)
    return np.asarray(values, dtype=dtype)

def create_metadata(types=[str, unicode, int, float, list]):
    """ return a dictionary of metadata with various types
    """
    meta = {}
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for i, t in enumerate(types):
        if t in (str, unicode): 
            val = t('some string')
        elif t in (list, np.ndarray):
            val = t([1,2])
        elif type(t) is str:
            val = np.array([0], dtype=t)[0]
        else:
            val = t() # will instantiate some default value

        meta[letters[i]] = val
    return meta

def create_dimarray(shape=(2,3), dtype=float, axis_dtypes=float, dims=None, seed=SEED):
     
    # create an array of desired type
    values = create_array(shape, dtype=dtype, seed=seed)

    # create axes
    if type(axis_dtypes) is not tuple:
        axis_dtypes  = (axis_dtypes ,) * len(shape)
    axes = [create_array_axis(s, dtype, seed=seed) for s, dtype in zip(shape, axis_dtypes)]

    # create metadata
    meta = create_metadata()

    return da.DimArray(values, axes, dims=dims, **meta)

# def create_dataset(seed=SEED, dtypes = ("float", "int","int32", "int64")):
def create_dataset(seed=SEED, dtypes = ("float", "int","int32", "int64", "str", "unicode")):
    # a = create_dimarray((2,3,2,1,1), float, (float, int, str, unicode, object), seed=seed)
    ds = da.Dataset()
    ds['many_axes'] = create_dimarray((2,3,2), float, (float, int, str), seed=seed) # test axis types
    ds['shared_axis'] = ds['many_axes'].ix[0] # shares axes with "many_axes"  # test shared axes

    # test 0, 1 and 2-D variables of several types
    for shp in [(), (3,), (2,3)]:
        dims = ["x{}_{}d".format(i, len(shp)) for i in range(len(shp))]
        for dtype in dtypes:
            vname = "{}_{}d".format(dtype, len(shp))
            ds[vname] =  da.DimArray(create_array(shp, dtype, seed=seed), dims=dims)

    return ds

def assert_equal_axes(actual, expected, metadata=True):
    assert isinstance(expected, da.Axis)
    assert isinstance(actual, da.Axis)
    assert actual == expected
    if metadata:
        assert_equal_metadata(actual.attrs, expected.attrs)
        assert np.all(actual.weights == expected.weights)

def assert_equal_dimarrays(actual, expected, metadata=True, approx=False):
    assert isinstance(expected, da.DimArray)
    assert isinstance(actual, da.DimArray)
    assert actual.shape == expected.shape
    assert actual.dims == expected.dims

    # check the values
    if approx or actual.dtype.kind == 'f' and anynan(actual.values): 
        assert_equal_values = assert_almost_equal
    else: 
        assert_equal_values = assert_equal
    assert_equal_values(actual.values, expected.values)

    # check the axes
    for actual_axis, expected_axis in zip(actual.axes, expected.axes):
        assert_equal_axes(actual_axis, expected_axis, metadata=metadata)

    # check the metadata
    if metadata:
        assert_equal_metadata(actual.attrs, expected.attrs)

def assert_equal_metadata(actual, expected):
    assert actual.keys() == expected.keys()
    for k in actual.keys():
        assert np.all(actual[k] == expected[k])

def assert_equal_datasets(actual, expected, metadata=True, approx=False):
    assert expected.keys() == actual.keys()
    for k in expected.keys():
        try:
            assert_equal_dimarrays(actual[k], expected[k], approx=approx, metadata=metadata)
        except:
            print actual[k]
            print expected[k]
            print k
            raise
    # check the metadata
    if metadata:
        assert_equal_metadata(actual.attrs, expected.attrs)
