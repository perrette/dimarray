""" String representation for various data objects
"""
from collections import OrderedDict as odict
import numpy as np
import json
import dimarray as da
from dimarray.config import get_option

def str_attrs(meta, indent=4):
    return "\n".join([" "*indent+"{}: {}".format(key, repr(meta[key])) for key  in meta.keys()])

def repr_attrs(meta):
    return "\n".join([" "*4+"{}: {}".format(key, repr(meta[key])) for key  in meta.keys()])
    # return repr(odict(zip(meta.keys(), meta.values())))
    # return repr({k:meta[k] for k in meta.keys()})
    # return json.dumps({k:meta[k] for k in meta.keys()}, sort_keys=True,indent=4,separators=(',', ': '))
    # return json.dumps(odict(zip(meta.keys(), meta.values())))

def repr_axis(self, metadata=False):
    dim = self.name
    size = self.size
    first, last = self._bounds()
    if self.dtype.kind == 'M':
        first, last = str(first), str(last)
    elif self.dtype.kind == 'f':
        pass
    else:
        first, last = repr(first), repr(last)
    # if getattr(self.dtype, 'kind', None) == 'f':
    repr_ = "{dim} ({size}): {first} to {last}".format(dim=dim, size=size, first=first, last=last)
    # else:
    #     repr_ = "{dim} ({size}): {first} to {last}".format(dim=dim, size=size, first=repr(first), last=repr(last))
    if metadata and len(self.attrs)>0:
        repr_ += "\n"+str_attrs(self.attrs)
    return repr_

def repr_axes(self, metadata=False):
    return "\n".join([ "{i} / {axis}".format(i=i, axis=repr_axis(ax, metadata=metadata)  )
                      for i, ax in enumerate(self)])

str_axes = repr_axes

def repr_dimarray_inline(self, metadata=False, name=None):
    " inline representation of a dimarray (without its name"
    if name is None and hasattr(self, 'name'):
        name = self.name
    dims = self.dims
    if len(dims) == 0:
        val = self.values[()]
        if val.ndim == 1 and val.size == 1:
            val = val[0]
        descr = repr(val)
    else:
        descr = repr(dims)

    repr_ = ": ".join([name, descr]) 
    if metadata and len(self.attrs)>0:
        repr_ += "\n"+str_attrs(self.attrs)
    return repr_

def repr_dimarray(self, metadata=False, lazy=False):
    header = self.__class__.__name__
    # lazy = not isinstance(self, da.DimArray))
    if lazy:
        header = header + ": "+repr(self.name)+" (%i"%self.size+")"

    else:
        header = self.__class__.__name__.lower() + ": " + stats_dimarray(self)

    lines = [header]

    # axes
    if self.ndim > 0:
        lines.append(repr_axes(self.axes, metadata=metadata))

    # metadata
    if metadata and len(self.attrs) > 0:
        lines.append("attributes:")
        lines.append(repr_attrs(self.attrs) )
        # lines.append(str_attrs(self.attrs, indent=8) )

    # the data itself
    if lazy:
        # line = "array(...)" if self.ndim > 0 else str(self[0])
        # line = self.name+("(...)" if self.ndim > 0 else repr((self[0],)))
        line = ""
    elif self.size > get_option('display.max'):
        line = "array(...)"
    else:
        line = repr(self.values)
    if line:
        lines.append(line)

    return "\n".join(lines)

str_dimarray = repr_dimarray

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
        try:
            # numeric types
            desc['min']=self.min(skipna=True) 
            desc['max']=self.max(skipna=True) 
        except:
            pass
        stats = ", ".join([k+':'+desc[k] for k in desc])
    return stats

def repr_dataset(self, metadata=False):
    # variable names
    # nms = [nm for nm in self.keys() if nm not in self.dims]
    nms = [nm for nm in self.keys()]

    # header
    if not isinstance(self, da.Dataset):
        header = self.__class__.__name__+" of %s variables (%s)" % (len(nms), self.nc.file_format)
    else:
        header = "Dataset of %s variables" % len(nms)
    if len(nms) == 1: header = header.replace('variables','variable')

    lines = []
    lines.append(header)
    
    # display dimensions name, size, first and last value
    if len(self.axes) > 0:
        if metadata:
            lines.append("")
            lines.append("//dimensions:")
        lines.append(repr_axes(self.axes, metadata=metadata))

    # display variables name, shape and dimensions
    if len(self.keys()) > 0:
        if metadata:
            lines.append("")
            lines.append("//variables:")
    for nm in nms:
        dims = self.dims
        line = repr_dimarray_inline(self[nm], metadata=metadata, name=nm)
        lines.append(line)

    # Global Meta"data
    if metadata and len(self.attrs) > 0:
    # if len(self.attrs) > 0:
        # lines.append("//global attributes:\n"+str_attrs(self.attrs))
        lines.append("")
        lines.append("//global attributes:")
        lines.append(repr_attrs(self.attrs))

    return "\n".join(lines)

str_dataset = repr_dataset
