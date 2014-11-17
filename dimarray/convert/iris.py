# This has been adapted from iris (iris/lib/iris/pandas.py)
# See dimarray/LICENCE for license information about iris
"""
Provide conversion to and from iris Cube
"""
from __future__ import absolute_import

# import datetime
# import netcdftime
import numpy as np
# import pandas

import iris
import iris.util
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from dimarray.geo import GeoArray, DimArray, Axis

#
# From dimarray to iris
#
def _add_iris_metadata(iris_obj, meta):
    try:
        iris_obj.units = meta.pop('units',iris_obj.units) # '1' is default
    except Exception as error:
        msg = "warning: "+error.message
        print(msg)
        #warnings.warn('invalid units')
    iris_obj.long_name = meta.pop('long_name',iris_obj.long_name)
    iris_obj.standard_name = meta.pop('standard_name',iris_obj.standard_name)
    iris_obj.var_name = meta.pop('name',iris_obj.var_name)

    for k in meta:
        try:
            iris_obj.attributes[k] = meta[k]
        except ValueError as error:
            msg = "Warning when converting dimarray to iris cube:\n"
            msg += error.message
            print(msg)

def as_cube(dim_array, copy=True):
    """
    Convert a dimarray array into an Iris cube.

    Parameters
    ----------
    dim_array : DimArray instance
    copy : Whether to make a copy of the data.
                      Defaults to True.

    Returns
    -------
    iris cube

    Examples
    --------
    >>> from dimarray.geo import GeoArray
    >>> a = GeoArray([[1.,2.,3.],[4.,5.,6.]], lat=[30., 70.], lon=[-50., 30., 110.])
    >>> a.units = 'meters'
    >>> a.name = 'myarray'
    >>> a.long_name = 'test array for iris conversion'
    >>> as_cube(a)
    <iris 'Cube' of test array for iris conversion / (meters) (latitude: 2; longitude: 3)>
    """
    # Make the copy work consistently across NumPy 1.6 and 1.7.
    # (When 1.7 takes a copy it preserves the C/Fortran ordering, but
    # 1.6 doesn't. Since we don't care about preserving the order we can
    # just force it back to C-order.)
    order = 'C' if copy else 'A'
    data = np.array(dim_array, copy=copy, order=order)
    cube = Cube(np.ma.masked_invalid(data, copy=False))

    # add coordinates
    for i, ax in enumerate(dim_array.axes):
        coord = as_coord(ax)

        if isinstance(coord, DimCoord):
            cube.add_dim_coord(coord, i)
        else:
            cube.add_aux_coord(coord, i)

    # add cube metadata
    _add_iris_metadata(cube, dim_array.attrs)

    return cube

def as_coord(ax):
    """ convert dimarray or geoarray axis to Cube coordinate
    """
    points = np.array(ax.values) # make a copy of array values
    name = ax.name
    if (np.issubdtype(points.dtype, np.number) and
            iris.util.monotonic(points, strict=True)):
        coord = DimCoord(points, var_name=name) #TODO: define circular based on modulo attribute?
        #coord.rename(name)

    else:
        coord = AuxCoord(points, var_name=name)
        #coord.rename(name)

    # add coordinate metadata
    _add_iris_metadata(coord, ax.attrs)

    return coord

#
# From iris to dimarray
#
def _add_dimarray_metadata(dim_obj, iris_obj):
    if iris_obj.standard_name: dim_obj.standard_name = iris_obj.standard_name
    if iris_obj.long_name: dim_obj.long_name = iris_obj.long_name
    if iris_obj.units: dim_obj.units = iris_obj.units.name
    if iris_obj.var_name: dim_obj.name = iris_obj.var_name
    dim_obj.__dict__.update(iris_obj.attributes)

def as_axis(coord):
    """ convert Cube coordinate to DimArray Axis
    """
    values = np.array(coord.points)
    name = coord.var_name
    ax = Axis(values, name)
    _add_dimarray_metadata(ax, coord)

    return ax

def as_dimarray(cube, copy=True, cls=None):
    """ convert Cube to GeoArray
    """
    values = cube.data
    axes = []
    # NOTE: There seems to be nothing in iris that 
    # prevents the number of dim_coords from
    # being different from the actual number of 
    # dimensions of array
    for coord in cube.coords(dim_coords=True):
        ax = as_axis(coord)
        axes.append(ax)

    if cls is None: cls = DimArray
    dim_array = cls(values, axes, copy=copy)

    _add_dimarray_metadata(dim_array, cube)

    return dim_array

def as_geoarray(cube, copy=True):
    return as_dimarray(cube, copy=copy, cls=GeoArray)
