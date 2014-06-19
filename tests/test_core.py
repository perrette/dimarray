import sys
from warnings import warn
import numpy as np
import dimarray as da

def doctest_indexing():
    """ Various indexing tests in addition to what's in the doc

Get Items:

>>> v = DimArray([[1,2],[3,4],[5,6],[7,8]], labels=[["a","b","c","d"], [10.,20.]], dims=['x0','x1'], dtype=float) 
>>> v
dimarray: 8 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (4): a to d
1 / x1 (2): 10.0 to 20.0
array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.],
       [ 7.,  8.]])
>>> v['a',20]  # extract a single item
2.0
>>> v.ix[0, 1] # or use `ix` to use integer indexing
2.0
>>> v['a':'c',10]  # 'c' is INCLUDED
dimarray: 3 non-null elements (0 null)
dimensions: 'x0'
0 / x0 (3): a to c
array([ 1.,  3.,  5.])
>>> v[['a','c'],10]  # it is possible to provide a list
dimarray: 2 non-null elements (0 null)
dimensions: 'x0'
0 / x0 (2): a to c
array([ 1.,  5.])
>>> v[v.x0 != 'b',10]  # boolean indexing is also fine
dimarray: 3 non-null elements (0 null)
dimensions: 'x0'
0 / x0 (3): a to d
array([ 1.,  5.,  7.])
>>> v[['a','c'],[10,20]]  # it is possible to provide a list
dimarray: 2 non-null elements (0 null)
dimensions: 'x0,x1'
0 / x0,x1 (2): ('a', '10.0') to ('c', '20.0')
array([ 1.,  6.])
>>> v.box[['a','c'],[10,20]]  # indexing on each dimension, individually
dimarray: 4 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (2): a to c
1 / x1 (2): 10.0 to 20.0
array([[ 1.,  2.],
       [ 5.,  6.]])

Set Items:
>>> v[:] = 0
>>> v
dimarray: 8 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (4): a to d
1 / x1 (2): 10.0 to 20.0
array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]])
>>> v['d'] = 1
>>> v
dimarray: 8 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (4): a to d
1 / x1 (2): 10.0 to 20.0
array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 1.,  1.]])
>>> v['b', 10] = 2
>>> v
dimarray: 8 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (4): a to d
1 / x1 (2): 10.0 to 20.0
array([[ 0.,  0.],
       [ 2.,  0.],
       [ 0.,  0.],
       [ 1.,  1.]])
>>> v.box[['a','c'],[10,20]] = 3
>>> v
dimarray: 8 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (4): a to d
1 / x1 (2): 10.0 to 20.0
array([[ 3.,  3.],
       [ 2.,  0.],
       [ 3.,  3.],
       [ 1.,  1.]])
>>> v[['a','c'],[10,20]] = 4
>>> v
dimarray: 8 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (4): a to d
1 / x1 (2): 10.0 to 20.0
array([[ 4.,  3.],
       [ 2.,  0.],
       [ 3.,  4.],
       [ 1.,  1.]])
>>> v.values[-1] = 5 # last element to 5 
>>> v
dimarray: 8 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (4): a to d
1 / x1 (2): 10.0 to 20.0
array([[ 4.,  3.],
       [ 2.,  0.],
       [ 3.,  4.],
       [ 5.,  5.]])
>>> v.ix[-1] = 6
>>> v
dimarray: 8 non-null elements (0 null)
dimensions: 'x0', 'x1'
0 / x0 (4): a to d
1 / x1 (2): 10.0 to 20.0
array([[ 4.,  3.],
       [ 2.,  0.],
       [ 3.,  4.],
       [ 6.,  6.]])
    """
    pass


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

def main(**kwargs):
    import pytest
    from doctesting import MyDocTest
    pytest.main() # unit test above
    MyDocTest(__name__).testmod()

if __name__ == "__main__":
    main()
