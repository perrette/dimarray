""" Plotting methods for dimarray class
"""
from __future__ import division
import numpy as np
from dimarray.tools import anynan

def _get_axis_labels(dim_axis):
    """ return values, xticklabels and label for an axis
    """
    if not dim_axis.is_numeric():
        values = np.arange(0, dim_axis.size/2, 0.5) # x-axis
        ticklabels = dim_axis.values
        ticks = values
    else:
        values = dim_axis.values
        ticklabels = None
        ticks = None

    # axis label for the plot
    label = dim_axis.name
    if hasattr(dim_axis, 'units'):
        label = label+' ({})'.format(dim_axis.units)

    return values, ticks, ticklabels, label

def _plot1D(self, funcname, *args, **kwargs):
    """ generic wrapper for 1-D plots (e.g. plot, scatter)
    """
    import matplotlib.pyplot as plt
    ndims = kwargs.pop('_ndims', None) # max dimension allowed
    if len(self.dims) > 2:
        raise NotImplementedError(funcname+" only works for 1-D or 2-D data")

    _transpose = kwargs.pop('_transpose', False)

    if 'ax' in kwargs: 
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()
    function = getattr(ax, funcname)

    # keyword argument to indicate legend, for 'plot'
    legend=kwargs.pop('legend', True)
    legend = legend and self.ndim==2

    xval, xticks, xticklab, xlab = _get_axis_labels(self.axes[0])
    if _transpose:
        pc = function(xval, self.values.T, *args, **kwargs)
    else:
        pc = function(xval, self.values, *args, **kwargs)
    # add labels
    if xlab is not None: ax.set_xlabel(xlab)
    if xticks is not None: ax.set_xticks(xticks)
    if xticklab is not None: ax.set_xticklabels(xticklab)

    # add y-labels?
    if hasattr(self, 'name') and hasattr(self, 'units') and self.name is not None and self.units is not None:
        lab = "{} ({})".format(self.name, self.units)
    elif hasattr(self, 'name') and self.name is not None:
        lab = "{}".format(self.name)
    elif hasattr(self, 'units') and self.units is not None:
        lab = "{}".format(self.units)
    else: 
        lab = None
    if lab is not None:
        ax.set_ylabel(lab)

    # add labels
    if self.ndim == 2:
        for i, line in enumerate(pc):
            line.set_label(str(self.axes[1].values[i]))

    # add legend
    if legend:
        ax.legend()

    return pc

def _plot2D(self, funcname, *args, **kwargs):
    """ generic plotting function for 2-D plots
    """
    if len(self.dims) != 2:
        raise NotImplementedError(funcname+" can only be called on two-dimensional dimarrays.")

    import matplotlib.pyplot as plt
    
    #ax = plt.gca()
    if 'ax' in kwargs: 
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()

    colorbar = kwargs.pop('colorbar', False)

    # get the actual plotting function
    function = getattr(ax, funcname)
    # extract information about axis value and labels
    # e.g. transform non-numeric data to float and
    # set appropriate labels afterwards.
    xval, xticks, xticklab, xlab = _get_axis_labels(self.axes[1])
    yval, yticks, yticklab, ylab = _get_axis_labels(self.axes[0])
    values = self.values

    # pcolor does not work with nans
    if (funcname == 'pcolormesh' or funcname == 'pcolor') and anynan(values):
        values = np.ma.array(values, mask=np.isnan(values))

    # make the plot
    pc = function(xval, yval, values, **kwargs)
    # add labels
    if xlab is not None: ax.set_xlabel(xlab)
    if ylab is not None: ax.set_ylabel(ylab)
    if xticklab is not None: ax.set_xticklabels(xticklab)
    if xticks is not None: ax.set_xticks(xticks)
    if yticklab is not None: ax.set_yticklabels(yticklab)
    if yticks is not None: ax.set_yticks(yticks)

    # add colorbar?
    if colorbar:
        plt.colorbar(pc, ax=ax)
    return pc

#
# 1-D functions
#
def plot(self, *args, **kwargs):
    """ Plot 1-D or 2-D data.
    
    Wraps matplotlib's plot()

    Parameters
    ----------
    *args, **kwargs : passed to matplotlib.pyplot.plot
    legend : True (default) or False
        Display legend for 2-D data.
    ax : matplotlib.Axis, optional
        Provide axis on which to show the plot.

    Returns
    -------
    lines : list of matplotlib's Lines2D instances

    Note
    ----
    You can use to_pandas().plot() combination to use pandas' plot method.

    Examples
    --------
    >>> from dimarray import DimArray
    >>> data = DimArray(np.random.rand(4,3), axes=[np.arange(4), ['a','b','c']], dims=['distance', 'label'])
    >>> data.axes[0].units = 'meters'
    >>> h = data.plot(linewidth=2)
    >>> h = data.T.plot(linestyle='-.')
    >>> h = data.plot(linestyle='-.', legend=False)
    """
    assert self.ndim <= 2, "only support plotting for 1- and 2-D objects"
    return self._plot1D('plot', *args, **kwargs)

def bar(self, *args, **kwargs):
    """ Make a bar plot

    Wraps matplotlib bar()
    See plot documentation in matplotlib for accepted keyword arguments.
    """
    if len(self.dims) > 1:
        raise NotImplementedError("plot can only be called up to one-dimensional array.")
    return self._plot1D('bar',*args, **kwargs)

def barh(self, *args, **kwargs):
    """ Make a horizontal bar plot

    Wraps matplotlib barh()
    See barh documentation in matplotlib for accepted keyword arguments.
    """
    if len(self.dims) > 1:
        raise NotImplementedError("plot can only be called up to two-dimensional array.")
    return self._plot1D('barh',*args, **kwargs)

def stackplot(self, *args, **kwargs):
    """ Draws a stacked area plot.

    Wraps matplotlib stackplot()
    See stackplot documentation in matplotlib for accepted keyword arguments.
    """
    if len(self.dims) > 2:
        raise NotImplementedError("plot can only be called up to two-dimensional dimarrays.")
    kwargs['_transpose'] = True
    kwargs['legend'] = False
    return self._plot1D('stackplot',*args, **kwargs)


#
# 2-D functions
#
def pcolor(self, *args, **kwargs):
    """ Plot a quadrilateral mesh. 
    
    Wraps matplotlib pcolormesh().
    See pcolormesh documentation in matplotlib for accepted keyword arguments.
    
    Examples
    --------
    >>> from dimarray import DimArray
    >>> x = DimArray(np.zeros([100,40]))
    >>> x.pcolor() # doctest: +SKIP
    >>> x.T.pcolor() # to flip horizontal/vertical axes  # doctest: +SKIP
    """
    if len(self.dims) != 2:
        raise NotImplementedError("pcolor can only be called on two-dimensional dimarrays.")

    return self._plot2D('pcolormesh', *args, **kwargs)

def contourf(self, *args, **kwargs):
    """ Plot filled 2-D contours. 
    
    Wraps matplotlib contourf().
    See contourf documentation in matplotlib for accepted keyword arguments.
    
    Examples
    --------
    >>> from dimarray import DimArray
    >>> x = DimArray(np.zeros([100,40])) 
    >>> x[:50,:20] = 1.
    >>> x.contourf() # doctest: +SKIP
    >>> x.T.contourf() # to flip horizontal/vertical axes  # doctest: +SKIP
    """
    if len(self.dims) != 2:
        raise NotImplementedError("contourf can only be called on two-dimensional dimarrays.")

    return self._plot2D('contourf', *args, **kwargs)

def contour(self, *args, **kwargs):
    """ Plot 2-D contours. 
    
    Wraps matplotlib contour().
    See contour documentation in matplotlib for accepted keyword arguments.
    
    Examples
    --------
    >>> from dimarray import DimArray
    >>> x = DimArray(np.zeros([100,40])) 
    >>> x[:50,:20] = 1.
    >>> x.contour() # doctest: +SKIP
    >>> x.T.contour() # to flip horizontal/vertical axes # doctest: +SKIP
    """
    if len(self.dims) != 2:
        raise NotImplementedError("contour can only be called on two-dimensional dimarrays.")

    return self._plot2D('contour', *args, **kwargs)
    

