""" Compatibility module which reproduces code from mpl_toolkits.basemap

copyright (c) 2011 by Jeffrey Whitaker.
Permission to use, copy, modify, and distribute this software and its documentation 
for any purpose and without fee is hereby granted, provided that the above copyright 
notices appear in all copies and that both the copyright notices and this permission 
notice appear in supporting documentation. THE AUTHOR DISCLAIMS ALL WARRANTIES WITH 
REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. 
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL 
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING 
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
import numpy as np

def interp(datain,xin,yin,xout,yout,checkbounds=False,masked=False,order=1):
    """
    Interpolate data (``datain``) on a rectilinear grid (with x = ``xin``
    y = ``yin``) to a grid with x = ``xout``, y= ``yout``.

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    datain           a rank-2 array with 1st dimension corresponding to
                     y, 2nd dimension x.
    xin, yin         rank-1 arrays containing x and y of
                     datain grid in increasing order.
    xout, yout       rank-2 arrays containing x and y of desired output grid.
    ==============   ====================================================

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    checkbounds      If True, values of xout and yout are checked to see
                     that they lie within the range specified by xin
                     and xin.
                     If False, and xout,yout are outside xin,yin,
                     interpolated values will be clipped to values on
                     boundary of input grid (xin,yin)
                     Default is False.
    masked           If True, points outside the range of xin and yin
                     are masked (in a masked array).
                     If masked is set to a number, then
                     points outside the range of xin and yin will be
                     set to that number. Default False.
    order            0 for nearest-neighbor interpolation, 1 for
                     bilinear interpolation, 3 for cublic spline
                     (default 1). order=3 requires scipy.ndimage.
    ==============   ====================================================

    .. note::
     If datain is a masked array and order=1 (bilinear interpolation) is
     used, elements of dataout will be masked if any of the four surrounding
     points in datain are masked.  To avoid this, do the interpolation in two
     passes, first with order=1 (producing dataout1), then with order=0
     (producing dataout2).  Then replace all the masked values in dataout1
     with the corresponding elements in dataout2 (using numpy.where).
     This effectively uses nearest neighbor interpolation if any of the
     four surrounding points in datain are masked, and bilinear interpolation
     otherwise.

    Returns ``dataout``, the interpolated data on the grid ``xout, yout``.
    """
    # xin and yin must be monotonically increasing.
    if xin[-1]-xin[0] < 0 or yin[-1]-yin[0] < 0:
        raise ValueError('xin and yin must be increasing!')
    if xout.shape != yout.shape:
        raise ValueError('xout and yout must have same shape!')
    # check that xout,yout are
    # within region defined by xin,yin.
    if checkbounds:
        if xout.min() < xin.min() or \
           xout.max() > xin.max() or \
           yout.min() < yin.min() or \
           yout.max() > yin.max():
            raise ValueError('yout or xout outside range of yin or xin')
    # compute grid coordinates of output grid.
    delx = xin[1:]-xin[0:-1]
    dely = yin[1:]-yin[0:-1]
    if max(delx)-min(delx) < 1.e-4 and max(dely)-min(dely) < 1.e-4:
        # regular input grid.
        xcoords = (len(xin)-1)*(xout-xin[0])/(xin[-1]-xin[0])
        ycoords = (len(yin)-1)*(yout-yin[0])/(yin[-1]-yin[0])
    else:
        # irregular (but still rectilinear) input grid.
        xoutflat = xout.flatten(); youtflat = yout.flatten()
        ix = (np.searchsorted(xin,xoutflat)-1).tolist()
        iy = (np.searchsorted(yin,youtflat)-1).tolist()
        xoutflat = xoutflat.tolist(); xin = xin.tolist()
        youtflat = youtflat.tolist(); yin = yin.tolist()
        xcoords = []; ycoords = []
        for n,i in enumerate(ix):
            if i < 0:
                xcoords.append(-1) # outside of range on xin (lower end)
            elif i >= len(xin)-1:
                xcoords.append(len(xin)) # outside range on upper end.
            else:
                xcoords.append(float(i)+(xoutflat[n]-xin[i])/(xin[i+1]-xin[i]))
        for m,j in enumerate(iy):
            if j < 0:
                ycoords.append(-1) # outside of range of yin (on lower end)
            elif j >= len(yin)-1:
                ycoords.append(len(yin)) # outside range on upper end
            else:
                ycoords.append(float(j)+(youtflat[m]-yin[j])/(yin[j+1]-yin[j]))
        xcoords = np.reshape(xcoords,xout.shape)
        ycoords = np.reshape(ycoords,yout.shape)
    # data outside range xin,yin will be clipped to
    # values on boundary.
    if masked:
        xmask = np.logical_or(np.less(xcoords,0),np.greater(xcoords,len(xin)-1))
        ymask = np.logical_or(np.less(ycoords,0),np.greater(ycoords,len(yin)-1))
        xymask = np.logical_or(xmask,ymask)
    xcoords = np.clip(xcoords,0,len(xin)-1)
    ycoords = np.clip(ycoords,0,len(yin)-1)
    # interpolate to output grid using bilinear interpolation.
    if order == 1:
        xi = xcoords.astype(np.int32)
        yi = ycoords.astype(np.int32)
        xip1 = xi+1
        yip1 = yi+1
        xip1 = np.clip(xip1,0,len(xin)-1)
        yip1 = np.clip(yip1,0,len(yin)-1)
        delx = xcoords-xi.astype(np.float32)
        dely = ycoords-yi.astype(np.float32)
        dataout = (1.-delx)*(1.-dely)*datain[yi,xi] + \
                  delx*dely*datain[yip1,xip1] + \
                  (1.-delx)*dely*datain[yip1,xi] + \
                  delx*(1.-dely)*datain[yi,xip1]
    elif order == 0:
        xcoordsi = np.around(xcoords).astype(np.int32)
        ycoordsi = np.around(ycoords).astype(np.int32)
        dataout = datain[ycoordsi,xcoordsi]
    elif order == 3:
        try:
            from scipy.ndimage import map_coordinates
        except ImportError:
            raise ValueError('scipy.ndimage must be installed if order=3')
        coords = [ycoords,xcoords]
        dataout = map_coordinates(datain,coords,order=3,mode='nearest')
    else:
        raise ValueError('order keyword must be 0, 1 or 3')
    if masked and isinstance(masked,bool):
        dataout = ma.masked_array(dataout)
        newmask = ma.mask_or(ma.getmask(dataout), xymask)
        dataout = ma.masked_array(dataout,mask=newmask)
    elif masked and is_scalar(masked):
        dataout = np.where(xymask,masked,dataout)
    return dataout

def shiftgrid(lon0,datain,lonsin,start=True,cyclic=360.0):
    """
    Shift global lat/lon grid east or west.

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    lon0             starting longitude for shifted grid
                     (ending longitude if start=False). lon0 must be on
                     input grid (within the range of lonsin).
    datain           original data.
    lonsin           original longitudes.
    ==============   ====================================================

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    start            if True, lon0 represents the starting longitude
                     of the new grid. if False, lon0 is the ending
                     longitude. Default True.
    cyclic           width of periodic domain (default 360)
    ==============   ====================================================

    returns ``dataout,lonsout`` (data and longitudes on shifted grid).
    """
    if np.fabs(lonsin[-1]-lonsin[0]-cyclic) > 1.e-4:
        # Use all data instead of raise ValueError, 'cyclic point not included'
        start_idx = 0
    else:
        # If cyclic, remove the duplicate point
        start_idx = 1
    if lon0 < lonsin[0] or lon0 > lonsin[-1]:
        raise ValueError('lon0 outside of range of lonsin')
    i0 = np.argmin(np.fabs(lonsin-lon0))
    i0_shift = len(lonsin)-i0
    if ma.isMA(datain):
        dataout  = ma.zeros(datain.shape,datain.dtype)
    else:
        dataout  = np.zeros(datain.shape,datain.dtype)
    if ma.isMA(lonsin):
        lonsout = ma.zeros(lonsin.shape,lonsin.dtype)
    else:
        lonsout = np.zeros(lonsin.shape,lonsin.dtype)
    if start:
        lonsout[0:i0_shift] = lonsin[i0:]
    else:
        lonsout[0:i0_shift] = lonsin[i0:]-cyclic
    if datain.ndim == 2:
        dataout[:,0:i0_shift] = datain[:,i0:]
    elif datain.ndim == 1:
        dataout[0:i0_shift] = datain[i0:]
    else:
        raise ValueError('data must be 1d or 2d with longitude as 2nd dim')
    if start:
        lonsout[i0_shift:] = lonsin[start_idx:i0+start_idx]+cyclic
    else:
        lonsout[i0_shift:] = lonsin[start_idx:i0+start_idx]
    if datain.ndim == 2:
       dataout[:,i0_shift:] = datain[:,start_idx:i0+start_idx]
    elif datain.ndim == 1:
       dataout[i0_shift:] = datain[start_idx:i0+start_idx]
    return dataout,lonsout
