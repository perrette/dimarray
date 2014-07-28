""" Plotting methods for dimarray class
"""
def plot(self, *args, **kwargs):
    """ by default, use pandas for plotting (for now at least)

    See doc at:

    help(pandas.Series.plot): 1-D

    help(pandas.DataFrame.plot): 2-D
    """
    assert self.ndim <= 2, "only support plotting for 1- and 2-D objects"
    return self.to_pandas().plot(*args, **kwargs)

def _plot2D(self, funcname, *args, **kwargs):
    """ generic plotting function for 2-D plots
    """
    if len(self.dims) != 2:
        raise NotImplementedError("pcolor can only be called on two-dimensional dimarrays.")

    import matplotlib.pyplot as plt
    
    #ax = plt.gca()
    if 'ax' in kwargs: 
        ax = kwargs.pop('ax')

    else:
        ax = plt.gca()

    function = getattr(ax, funcname)
    pc = function(self.labels[1], self.labels[0], self.values, **kwargs)
    ax.set_xlabel(self.dims[1])
    ax.set_ylabel(self.dims[0])

    return pc

def pcolor(self, *args, **kwargs):
    """ Plot a quadrilateral mesh. 
    
    Wraps matplotlib pcolormesh().
    See pcolormesh documentation in matplotlib for accepted keyword arguments.
    
    Examples
    --------
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
    >>> x = DimArray(np.zeros([100,40])) 
    >>> x[:50,:20] = 1.
    >>> x.contour() # doctest: +SKIP
    >>> x.T.contour() # to flip horizontal/vertical axes # doctest: +SKIP
    """
    if len(self.dims) != 2:
        raise NotImplementedError("contour can only be called on two-dimensional dimarrays.")

    return self._plot2D('contour', *args, **kwargs)
    
