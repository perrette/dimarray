import sys
from dimarray.testing import testmod

def indexing():
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


def main(**kwargs):

    import metadata as metadata
    import dimarraycls
    import axes as axes
    import _indexing as indexing
    import _reshape as reshape
    import _transform as transform
    import missingvalues as missingvalues 
    import _operation as operation
    import align as align

    testmod(metadata, **kwargs)
    testmod(dimarraycls, **kwargs)
    testmod(axes, **kwargs)
    testmod(indexing, **kwargs)
    testmod(transform, **kwargs)
    testmod(reshape, **kwargs)
    testmod(missingvalues, **kwargs)
    testmod(operation, **kwargs)
    testmod(align, **kwargs)
    testmod(sys.modules[__name__], **kwargs)

if __name__ == "__main__":
    main()
