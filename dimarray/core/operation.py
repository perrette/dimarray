"""
Operation and axis aligmnent
"""
import numpy as np

from align import align_dims, align_axes
from tools import is_DimArray

def operation(func, o1, o2, reindex=True, broadcast=True, constructor=None):
    """ operation on LaxArray objects

    Parameters
    ----------
        func        : operator
        o1            : LHS operand: DimArray
        o2            : RHS operand: at least: be convertible by np.array())
        align, optional: if True, use pandas to align the axes

    Returns
    -------
        values: array values
        dims : dimension names
    """
    if constructor is None:
        constructor = o1._constructor

    # second operand is not a DimArray: let numpy do the job 
    if not is_DimArray(o2): # isinstance
        if np.ndim(o2) > np.ndim(o1):
            raise ValueError("bad input: second operand's dimensions not documented")
        res = func(o1.values, np.array(o2))
        return constructor(res, o1.axes)

    # check for first operand (reverse operations)
    elif not is_DimArray(o1): # isinstance
        if np.ndim(o1) > np.ndim(o2):
            raise ValueError("bad input: second operand's dimensions not documented")
        res = func(np.array(o1), o2.values)
        return constructor(res, o2.axes)

    # both objects are dimarrays

    # Align axes by re-indexing
    if reindex:
        o1, o2 = align_axes(o1, o2)

    # Align dimensions by adding new axes and transposing if necessary
    if broadcast:
        o1, o2 = align_dims(o1, o2)

    # make the new axes
    newaxes = o1.axes.copy()

    # ...make sure no singleton value is included
    for i, ax in enumerate(newaxes):
        if ax.values[0] is None:
            newaxes[i] = o2.axes[ax.name]

    res = func(o1.values, o2.values)

    return constructor(res, newaxes)
