""" Provide example datasets to test the module
"""
import os
import numpy as np

from geotools import Dimarray, Map, TimeSeries, Axes, Axis, _ncio
if _ncio:
    from geotools import ncio as nc

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, "data")

def get_data_1D():
    ts = get_timeseries()
    return ts.time, ts.values

def get_data_2D():
    """ example 2D dataset as numpy arrays
    """
    if _ncio:
	datadir = "/home/perrette/Projects/SLR_PrimapLive/code/core/emulator/data"
	datafile = os.path.join(datadir, "emulator.oceandyn.gmtscaling.nc")
	data = nc.read(datafile,"CSIRO-Mk3-6-0")
	lon = data.lon
	lat = data.lat
	data = data.values

    else:
	lon = np.linspace(0.5,359.5,360)
	lat = np.linspace(-89.5,89.5,180)
	lon2, lat2 = np.meshgrid(lon, lat)
	data = np.cos(np.radians(lat2))*np.sin(np.radians(lon2)*5)

    return lat, lon, data

def make_data(nlat=180, nlon=360):
     """ generate some example data
     """
     np.random.seed(0) # just for reproductivity of the numbers !
     values = np.random.randn(nlat, nlon)
     lon = np.linspace(-179.5,179.5,nlon)
     lat = np.linspace(-89.5,89.5,nlat)
     lon2, lat2 = np.meshgrid(lon, lat)
     data = np.cos(np.radians(lat2))*np.sin(np.radians(lon2)*5)
     return lon, lat, values


#
# geotools objects
#

def get_axes(*names):
    """
    """
    time = np.arange(1850,2101)
    items = ["antarctica","greenland"]
    lon = np.linspace(0.5,359.5,360)
    lat = np.linspace(-89.5,89.5,180)

    ax = Axes()
    ax.append(time,"time")
    ax.append(lon,"lon")
    ax.append(lat,"lat")
    ax.append(items,"items")

    if len(names) >0:
	ax = ax[names]

    return ax

def get_map():
    """ return a map (useful for testing)
    """
    lat, lon, data = get_data_2D()
    return Map(data, lat, lon)

def get_timeseries():
    """ return a time series
    """
    yr = np.arange(1900,2100)
    total = (((yr-1900)/150.)**2).cumsum()
    return TimeSeries(total, yr)

