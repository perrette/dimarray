""" Test vector projection
"""
import unittest
import numpy as np
import dimarray as da
from dimarray.geo import transform_vectors
from dimarray.geo.crs import get_crs

# Test: compute flowline based on an idealized vector field, and check whether 
# it is conserved upon transformation.


# class TestPolarStereographicData(unittest.TestCase):
def this_is_messy_fix_it():
    """This is a manual, visual test for vector coordinate transform.
    Should think of something more robust, to be handled with py.test
    """

    from scipy.interpolate import interp2d

    map0 = {'ellipsoid': u'WGS84',
     'false_easting': 0.0,
     'false_northing': 0.0,
     'grid_mapping_name': u'polar_stereographic',
     'latitude_of_projection_origin': 90.0,
     'standard_parallel': 71.0,
     'straight_vertical_longitude_from_pole': -39.0}

    map1 = {'ellipsoid': u'WGS84',
     'false_easting': 0.0,
     'false_northing': 0.0,
     'grid_mapping_name': u'polar_stereographic',
     'latitude_of_projection_origin': 90.0,
     'standard_parallel': 60.0,
     'straight_vertical_longitude_from_pole': -50.0}

    # def test_drift(self):
    ny, nx = 100, 120
    x = np.linspace(-800, 800, nx)
    y = np.linspace(-3000, -1000, ny)
    vx = da.ones(axes=[('y',y),('x',x)])
    vy = da.ones(axes=vx.axes)

    def compute_drift(x0, y0, x, y, vx, vy, n=100, s=1):
        """ Compute drift from a starting point
        """
        # prepare interpolation
        fx = interp2d(x, y, vx)
        fy = interp2d(x, y, vy)

        line = [(x0, y0)]
        for i in range(n):
            x0 += fx(x0, y0)
            y0 += fy(x0, y0)
            line.append((x0,y0))
        xx, yy = zip(*line)
        return np.array(xx), np.array(yy)

    # drift for the standard coordinate system
    x0, y0 = 0, -2000
    xx, yy = compute_drift(x0, y0, vx.x, vx.y, vx.values, vy.values)
    xend, yend = xx[-1], yy[-1] # end point

    # drift after transformation
    crs0 = get_crs(map0)
    crs1 = get_crs(map1)

    # transform vector and start point
    missing = -99
    vxt,vyt = transform_vectors(vx,vy, from_crs=crs0, to_crs=crs1, masked=missing)
    # X, Y = np.meshgrid(vx.x, vx.y)
    # Xt,Yt,_ = crs1.transform_points(crs0, X, Y).T
    # vxt,vyt = crs1.transform_vectors(crs0, X, Y, vx.values,vy.values)
    x0t, y0t = crs1.transform_point(x0, y0, crs0)


    xxt, yyt = compute_drift(x0t, y0t, vxt.x, vxt.y, vxt.values, vyt.values)
    # xxt, yyt = compute_drift(x0t, y0t, Xt, Yt, vxt, vyt)
    xxtt, yytt,_ = crs0.transform_points(crs1, xxt, yyt).T # transform back
    xtend, ytend = xxtt[-1], yytt[-1] # end point

    # Check plot
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure(1)
    vxt[vxt==missing] = np.nan
    vyt[vyt==missing] = np.nan
    plt.streamplot(vx.x, vx.y, vx.to_MaskedArray(),vy.to_MaskedArray())
    plt.streamplot(vxt.x, vxt.y, vxt.to_MaskedArray(), vy.to_MaskedArray())
    plt.figure(2)
    plt.plot(xx, yy, label='truth')
    plt.plot(xxt, yyt, ls=':', label='temporary transformed')
    plt.plot(xxtt, yytt, ls='--', label='transformed back and forth')
    plt.scatter(xend, yend,color='b')
    plt.scatter(xtend, ytend,color='g')
    plt.legend(frameon=False)
    #plt.draw()
    plt.show()

if __name__ == '__main__':
    this_is_messy_fix_it()
