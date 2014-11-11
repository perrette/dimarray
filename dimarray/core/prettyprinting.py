""" String representation for various data objects
"""
from collections import OrderedDict as odict
import numpy as np
import dimarray as da
from dimarray.config import get_option

def repr_attrs(meta):
    return "\n".join([" "*4+"{}: {}".format(key, meta[key]) for key  in meta.keys()])

def repr_axis(self, metadata=False):
    dim = self.name
    size = self.size
    first, last = self.range()
    repr_ = "{dim} ({size}): {first} to {last}".format(**locals())
    if metadata:
        repr_ += "\n"+repr_attrs(self.attrs)
    return repr_

def repr_axes(self, metadata=False):
    return "\n".join([ "{i} / {axis}".format(i=i, 
                                             axis=repr_axis(self[dim]), 
                                             metadata=metadata)
                      for i, dim in enumerate(self.dims)])


def repr_dimarray_inline(self, metadata=True, name=None):
    " inline representation of a dimarray (without its name"
    if name is None and hasattr(self, 'name'):
        name = self.name
    dims = self.dims
    descr = ",".join(dims) if len(dims) > 0 else str(self[0])
    return ": ".join([name, descr]) 

def repr_dimarray(self, metadata=False, lazy=False, backcompatibility=True):
    var = self.nc.variables[self.name]
    dims = var.dimensions
    name = self.name
    f = self.nc
    
    # header
    if backcompatibility and \
            (isinstance(self, da.Dataset) or isinstance(self, da.DimArray)):
        header = self.__class__.__name__.lower()
    else:
        header = self.__class__.__name__

    # add various levels of description
    if hasattr(self, 'name'):
        descr1 = self.name
    else:
        descr1 = ""
    if lazy:
        descr2 = "on disk"
    else:
        descr2 = stats_dimarray(self, backcompatibility=backcompatibility)

    if descr1 != "":
        header += ": "+descr1
    if descr2 != "":
        header += " ("+descr2+")"

    lines = [header]

    # axes
    if self.ndim > 0:
        lines.append(repr_axes(self.axes, metadata=metadata))

    # metadata
    if metadata and len(self.attrs) > 0:
        lines.append( repr_attrs(self.attrs) )

    # the data itself
    if lazy:
        line = "array(...)" if self.ndim > 0 else str(self[0])
    elif self.size > get_option('display.max'):
        line = "array(...)"
    else:
        line = repr(self.values)
    lines.append(line)

    return "\n".join(lines)

def stats_dimarray(self, backcompatibility=True):
    """ descriptive statistics
    """
    try:
        if self.ndim > 0:
            nonnull = np.size(self.values[~np.isnan(self.values)])
        else:
            nonnull = int(~np.isnan(self.values))

    except TypeError: # e.g. object
        nonnull = self.size

    if backcompatibility:
        stats = "{} non-null elements ({} null)".format(nonnull, self.size-nonnull)
    else:
        desc = odict()
        if nonnull < self.size:
            desc['nans']=self.size-nonnull
        desc['min']=self.min(skipna=True) 
        desc['max']=self.max(skipna=True) 
        stats = ", ".join([k+':'+desc[k]])
    return stats

def repr_dataset(self, metadata=True):
    # variable names
    nms = [nm for nm in self.keys() if nm not in self.dims]

    # header
    header = "Dataset of %s variables (netCDF)" % (len(nms))
    if len(nms) == 1: header = header.replace('variables','variable')

    lines = []
    lines.append(header)
    
    # display dimensions name, size, first and last value
    lines.append(repr_axes(self.axes, metadata=metadata))

    # display variables name, shape and dimensions
    for nm in nms:
        dims = self.dims
        line = repr_dimarray_inline(self[nm], metadata=metadata, name=nm)
        lines.append(line)

    # Meta"data
    if metadata and len(self.attrs) > 0:
        lines.append("//global attributes:\n"+repr_attrs(self.attrs))

    return "\n".join(lines)
