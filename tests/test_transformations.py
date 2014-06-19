import sys
from warnings import warn
import numpy as np
import dimarray as da

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
