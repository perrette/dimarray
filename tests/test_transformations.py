import sys
from warnings import warn
import numpy as np
import dimarray as da
import pytest


@pytest.fixture()
def v():
    import numpy as np
    values = np.arange(4*3).reshape(4,3)
    time = 'time', np.arange(1950,1954) 
    lat = 'lat', np.linspace(-90,90,3)
    return da.DimArray(values, axes=[np.arange(1950,1954), np.linspace(-90,90,3)], dims=['time','lat'])

#@pytest.fixture()
#def axis():
#    return None

def pytest_generate_tests(metafunc):
    # e.g. test transformations with 3 axis values
    if 'axis' in metafunc.fixturenames:
        metafunc.parametrize("axis", [None, 'time', 'lat'])
        #metafunc.parametrize("axis", [None, 0, 1])

def test_all(v, axis):

    axis_id, axis_nm = v._get_axis_info(axis)

    # sum, product
    assert np.all(v.sum(axis=axis_nm) == v.values.sum(axis=axis_id))
    assert np.all(v.prod(axis=axis_nm) == v.values.prod(axis=axis_id))

    # moments
    assert np.all(v.mean(axis=axis_nm) == v.values.mean(axis=axis_id))
    assert np.all(v.var(axis=axis_nm) == v.values.var(axis=axis_id))
    assert np.all(v.std(axis=axis_nm) == v.values.std(axis=axis_id))

    # median, min, max, peak-to-peak
    assert np.all(v.median(axis=axis_nm) == np.median(v.values, axis=axis_id))
    assert np.all(v.min(axis=axis_nm) == v.values.min(axis=axis_id))
    assert np.all(v.max(axis=axis_nm) == v.values.max(axis=axis_id))
    assert np.all(v.ptp(axis=axis_nm) == v.values.ptp(axis=axis_id))

    # determine if all or any of the elements are True (or non-zero)
    assert np.all(v.all(axis=axis_nm) == v.values.all(axis=axis_id))
    assert np.all(v.any(axis=axis_nm) == v.values.any(axis=axis_id))

    # argmin, argmax: locate the minimum and maximum of an array
    if axis is None:
        assert np.all(v[v.argmin()] == v.min())
        assert np.all(v[v.argmax()] == v.max())
    else:
        v.argmin(axis=axis_nm)
        v.argmax(axis=axis_nm)

def test_transform():
    test_diff()
    try:
        test_vs_pandas()
    except ImportError:
        warn("pandas not installed, can't test transform against pandas")
    except AssertionError, msg:
        warn("pandas test failed {}".format(msg))

def test_diff():
    np.random.seed(0)
    v = da.DimArray(np.random.randn(5,7), {'time':np.arange(1950,1955), 'lat':np.linspace(-90,90,7)})
    v.diff(axis='time', keepaxis=False)
    v.diff(axis=0, keepaxis=False, scheme='centered')
    v.diff(axis=0, keepaxis=False, scheme='backward')
    v.diff(axis=0, keepaxis=False, scheme='forward')
    v.diff(axis=0, keepaxis=True, scheme='backward')
    v.diff(axis=0, keepaxis=True, scheme='forward')
    v.diff(n=2,axis=('time'), scheme='centered')

def _eq(x, y, tol=1e-6):
    """ test equality between two arrays, allowing for NaN and single precision
    """
    return np.all(np.isnan(x) | (np.abs(x-y) < tol))

def test_vs_pandas():
    np.random.seed(0)
    v = da.DimArray(np.random.randn(5,7), {'time':np.arange(1950,1955), 'lat':np.linspace(-90,90,7)})
    assert _eq(v.std(ddof=1, axis=0).values, v.to_pandas().std().values), "std vs pandas"
    assert _eq(v.var(ddof=1, axis=0).values, v.to_pandas().var().values), "var vs pandas"
    assert _eq(v.cumsum(axis=0).values , v.to_pandas().cumsum().values), "pandas: cumsum failed"
    assert _eq(v.cumprod(axis=0).values , v.to_pandas().cumprod().values), "pandas: cumprod failed"
    assert _eq(v.diff(axis=0, keepaxis=True).cumsum(axis=0, skipna=True).values, v.to_pandas().diff().cumsum().values), "diff-cumsum failed"

    # TEST diff
    res = v.diff(axis=0, keepaxis=True) 
    assert _eq(res, v.to_pandas().diff()), 'diff failed'

def test_operations():
    a = da.DimArray([[1,2,3],[3,4,5]],dims=('x0','x1'))
    assert np.all(a == a)
    assert np.all(a+2 == a + np.ones(a.shape)*2)
    assert np.all(a+a == a*2)
    assert np.all(a*a == a**2)
    assert np.all((a - a.values) == a - a)

#def main(**kwargs):
#    import pytest
#    from testing import MyDocTest
#    pytest.main() # unit test above
#    MyDocTest(__name__).testmod()
#
#if __name__ == "__main__":
#    main()
