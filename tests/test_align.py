import numpy as np
from dimarray import DimArray
from dimarray.testing import assert_equal_dimarrays
from dimarray import align

def test_align():
    " test slices in an increasing axis"
    a = DimArray([1, 2, 3, 4], axes=[[1, 2, 3, 4]], dims=['dim0'])
    b = DimArray([0, 2, 4, 6], axes=[[0, 2, 4, 6]], dims=['dim0'])

    # outer join
    # expected
    a2 = DimArray([np.nan, 1, 2, 3, 4, np.nan], axes=[[0, 1, 2, 3, 4, 6]], dims=['dim0'])
    b2 = DimArray([0, np.nan, 2, np.nan, 4, 6], axes=[[0, 1, 2, 3, 4, 6]], dims=['dim0'])
    # got
    a2_got, b2_got = align(a, b, join="outer")
    # check
    assert_equal_dimarrays(a2, a2_got)
    assert_equal_dimarrays(b2, b2_got)

    # inner join
    # expected
    a3 = DimArray([2, 4], axes=[[2,4]], dims=['dim0'])
    b3 = DimArray([2, 4], axes=[[2,4]], dims=['dim0'])
    # got
    a3_got, b3_got = align(a, b, join="inner")
    # check
    assert_equal_dimarrays(a3, a3_got)
    assert_equal_dimarrays(b3, b3_got)
