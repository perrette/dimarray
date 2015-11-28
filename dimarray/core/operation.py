"""
Operation and axis aligmnent
"""
from __future__ import absolute_import
import numpy as np
import warnings

from dimarray.tools import is_DimArray
from .align import  align_dims, align as align_axes
from .axes import Axes

def operation(func, o1, o2, reindex=True, broadcast=True, constructor=None):
    """ binary operation involving a DimArray objects

    Parameters
    ----------
    func : operator
    o1 : LHS operand: DimArray
    o2 : RHS operand: at least: be convertible by np.array())
    align : bool, optional
        if True, use pandas to align the axes
    constructor : class Constructor, optional
        if None, o1's class constructor (o1._constructor) is used instead

    Returns
    -------
    DimArray instance
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

    # check grid mapping and emit a warning if mismatch
    if hasattr(o1, 'grid_mapping') and hasattr(o2, 'grid_mapping') \
            and o1.grid_mapping != o2.grid_mapping:
                warnings.warn("binary op : grid mappings mismatch")

    # Align axes by re-indexing
    if reindex:
        o1, o2 = align_axes((o1, o2))

    # Align dimensions by adding new axes and transposing if necessary
    if broadcast:
        o1, o2 = align_dims(o1, o2)

    # make the new axes
    newaxes = Axes()
    for i, ax in enumerate(o1.axes):
        if ax.values[0] is None:
            # ...make sure no singleton value is included
            newaxes.append(o2.axes[ax.name].copy())
        else:
            newaxes.append(ax.copy())

    res = func(o1.values, o2.values)

    return constructor(res, newaxes)
