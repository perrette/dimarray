""" Filling / interpolation
"""
import numpy as np

from axes import Axes, Axis

#
# linear interpolation
#

def interp(obj, indices, axis=0, method="linear", repna=False):
    """ 1-D interpolation of a DimArray along one or several (sequentially) axes

    Parameters
    ----------
    indices: array-like or dict
    axis, optional : axis name or position
    method: "nearest", "linear"
    repna: if True, replace out-of-bound values by NaN instead of raising an error

    Returns
    -------
    DimArray
    """
    kw = obj._get_keyword_indices(indices, axis)
    for k in kw:
        if method == "nearest":
            obj = _interp_nearest(obj, kw[k], k, repna=repna)
        else:
            obj = _interp_linear(obj, kw[k], k, repna=repna)
    return obj

def _interp_nearest(obj, values, axis, repna):
    """ "nearest" neighbour interpolation
    """
    ax = obj.axes[axis]
    pos = obj.dims.index(ax.name)
    assert ax.dtype is not np.dtype('O'), "interpolation only for non-object types"

    indices = np.zeros_like(values, dtype=int)
    mask = np.zeros_like(values, dtype=bool)

    for i, x in enumerate(values):
        res = _locate_nearest(ax, x)
        if res is None:
            if repna:
                mask[i] = True
                continue
            else:
                raise IndexError("value not found: {}".format(x))
            continue

        indices[i], _ = res

    # sample nearest neighbors
    result = obj.take(indices, axis=pos, indexing="position")
    result.put(np.nan, np.where(mask)[0], axis=pos, indexing="position", convert=True, inplace=True)
    result.axes[pos] = Axis(values, ax.name) # update axis

    return result

def _interp_linear(obj, newindices, axis, repna):
    """ linearly interpolate a dimarray along an axis
    """
    ax = obj.axes[axis]
    pos = obj.dims.index(ax.name)
    assert ax.dtype is not np.dtype('O'), "interpolation only for non-object types"

    i0 = np.zeros_like(newindices, dtype=int)
    i1 = np.zeros_like(newindices, dtype=int)
    w1 = np.empty_like(newindices, dtype=float)
    w1.fill(np.nan)

    for i, x in enumerate(newindices):
        res = _locate_bounds(ax, x)
        if res is None:
            if repna:
                continue
            else:
                raise IndexError("value not found: {}".format(x))
            continue

        i0[i], i1[i], w1[i] = res


    # sample nearest neighbors
    v0 = obj.take(i0, axis=pos, indexing="position")
    v1 = obj.take(i1, axis=pos, indexing="position")

    # result as weighted sum
    if not hasattr(v0, 'values'): # scalar
        return v0*(1-w1) + v1*w1
    else:
        newvalues = v0.values*(1-w1) + v1.values*w1

    axes = obj.axes.copy()
    axes[pos] = Axis(newindices, ax.name) # new axis
    return obj._constructor(newvalues, axes, **obj._metadata)


def _locate_nearest(axis, x):
    # index of nearest neighbour

    min, max = axis.values.min(), axis.values.max()
    if x > max or x < min: 
        return None

    i = axis.loc(x, tol=np.inf) # nearest neighbour search

    if i is None: 
        return None

    # out of bounds check
    xi = axis.values[i]

    return i, xi

def _locate_bounds(axis, x):
    """ return bounds around a values for interpolation
    """
    #assert is_regular(self.values), "interp mode only makes sense for regular axes !"

    #if x == 2: 1/0

    res = _locate_nearest(axis, x)

    if res is None:
        return None
    
    i, xi = res

    # make sure we have x in [xi, xi+1]
    if xi == x:
        return i, i, 0
    elif xi < x:
        i0, i1 = i, i+1
        x0, x1 = xi, axis.values[i1]
    else:
        i0, i1 = i-1, i
        x0, x1 = axis.values[i0], xi

    assert x0 <= x <= x1, "irregular axis, cannot interpolate !"

    # weight for interpolation
    w1 = (x-x0)/float(x1-x0)

    return i0, i1, w1
