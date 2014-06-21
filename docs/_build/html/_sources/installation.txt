Installation
------------

Download latest version on GitHub
=================================
https://github.com/perrette/dimarray/

Requires
========
- python2.7

- numpy

- netCDF4 (optional) :  for netCDF I/O
  
    repository: https://github.com/Unidata/netcdf4-python

    From source:
        the documentation can be found there: http://unidata.github.io/netcdf4-python/
        Basically you need to install HDF5 and netCDF4 libraries on your system before
        using pip or your favorite package manager.
    
    This can be annoying to install HDF5 and netCDF4 from source.
    Using the anaconda package from continuum analytics save time 
    (That is the only one I tried, but it possibly also 
    works with Enthought, xyPython or some other pre-compiled version of python)
    With conda (the package manager shipped with - but kind of independent from - anaconda) 
    it is enough to do a simple:

        conda install netCDF4 

- matplotlib (optional) : for plotting (for now plot command also requires pandas)

- pandas (optional) :  to_pandas() and from_pandas() methods, plot()

Installation
============

Download the latest version from github and extract from archive
Then from the dimarray repository type:
    
        python setup.py install  

Alternatively, you can use pip to download and install the version from pypi (could be slightly out-of-date):

        pip install dimarray 

or with conda:
    
        conda install dimarray

