""" This file is copied and adapted from cartopy. 

It is designed to be used in place of cartopy.crs

It is copied here until (and if) the pull request fix_issues_339_455
(commit b9211b6c7f072282818fdc527deccd9bcaa3b5ab)
has been merged into the master and included in the next release.
"""
from __future__ import absolute_import
import warnings
import numpy as np
from cartopy.crs import *
from cartopy.crs import (sgeom, NorthPolarStereo, SouthPolarStereo, Stereographic)
                         #RotatedPole, TransverseMercator, Globe, Geodetic, CRS


class FixedStereographic(Stereographic):
    def __init__(self, central_latitude=0.0, central_longitude=0.0,
                 false_easting=0.0, false_northing=0.0,
                 true_scale_latitude=None, 
                 scale_factor=None, # equivalent to 1.0
                 globe=None):
        proj4_params = [('proj', 'stere'), ('lat_0', central_latitude),
                        ('lon_0', central_longitude),
                        ('x_0', false_easting), ('y_0', false_northing)]

        if true_scale_latitude:
            if central_latitude not in (-90., 90.):
                warnings.warn('"true_scale_latitude" parameter is only used for polar stereographic projections. Consider the use of "scale_factor" instead.')
            proj4_params.append(('lat_ts', true_scale_latitude))

        # See https://github.com/SciTools/cartopy/issues/455
        if scale_factor:
            if true_scale_latitude is not None:
                warnings.warn('It does not make sense to provide both "scale_factor" and "true_scale_latitude"')
            proj4_params.append(('k_0', scale_factor))

        CRS.__init__(self, proj4_params, globe=globe)

        # TODO: Factor this out, particularly if there are other places using
        # it (currently: Stereographic & Geostationary). (#340)
        def ellipse(semimajor=2, semiminor=1, easting=0, northing=0, n=200):
            t = np.linspace(0, 2 * np.pi, n)
            coords = np.vstack([semimajor * np.cos(t), semiminor * np.sin(t)])
            coords += ([easting], [northing])
            return coords

        # TODO: Let the globe return the semimajor axis always.
        a = np.float(self.globe.semimajor_axis or 6378137.0)
        b = np.float(self.globe.semiminor_axis or 6356752.3142)

        # Note: The magic number has been picked to maintain consistent
        # behaviour with a wgs84 globe. There is no guarantee that the scaling
        # should even be linear.
        x_axis_offset = 5e7 / 6378137.
        y_axis_offset = 5e7 / 6356752.3142
        self._x_limits = (-a * x_axis_offset + false_easting,
                          a * x_axis_offset + false_easting)
        self._y_limits = (-b * y_axis_offset + false_northing,
                          b * y_axis_offset + false_northing)
        if self._x_limits[1] == self._y_limits[1]:
            point = sgeom.Point(false_easting, false_northing)
            self._boundary = point.buffer(self._x_limits[1]).exterior
        else:
            coords = ellipse(self._x_limits[1], self._y_limits[1],
                             false_easting, false_northing, 90)
            coords = tuple(tuple(pair) for pair in coords.T)
            self._boundary = sgeom.polygon.LinearRing(coords)
        self._threshold = np.diff(self._x_limits)[0] * 1e-3

    @property
    def boundary(self):
        return self._boundary

    @property
    def threshold(self):
        return self._threshold

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits


class FixedNorthPolarStereo(FixedStereographic):
    def __init__(self, central_longitude=0.0, true_scale_latitude=None, globe=None):
        FixedStereographic.__init__(self,
            central_latitude=90,
            central_longitude=central_longitude, 
            true_scale_latitude=true_scale_latitude, # None is equivalent to +90
            globe=globe)


class FixedSouthPolarStereo(FixedStereographic):
    def __init__(self, central_longitude=0.0, true_scale_latitude=None, globe=None):
        FixedStereographic.__init__(self, 
            central_latitude=-90,
            central_longitude=central_longitude, 
            true_scale_latitude=true_scale_latitude, # None is equivalent to -90
            globe=globe)


def _apply_fix(version_fixed=999):
    """Replace Cartopy version with their fixes.
    """
    import cartopy
    from types import ModuleType
    if not isinstance(cartopy, ModuleType):
        return # readthedocs's Mock
    M = cartopy.__version__.split('.')[0]
    m = cartopy.__version__.split('.')[1]
    if int(M) == 0 and int(m) < 11:
        warnings.warn('Projections were only tested for cartopy versions 0.11.x')
    return int(M) < version_fixed


if _apply_fix():
    Stereographic = FixedStereographic
    NorthPolarStereo = FixedNorthPolarStereo
    SouthPolarStereo = FixedSouthPolarStereo
