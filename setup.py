#from distutils.core import setup
from setuptools import setup

with open('README.rst') as file:
        long_description = file.read()

setup(name='dimarray',
      version='0.1.1',
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='array with labelled dimensions and axes, metadata and NaN handling',
      keywords=('labelled array','numpy','larry','pandas','iris'),
      packages = ['dimarray','dimarray.core','dimarray.geo','dimarray.io','dimarray.lib'],
      package_data = {
	  "dimarray": ['README.rst','dimarray.ipynb']
	  },
      long_description=long_description,
      url='https://github.com/perrette/dimarray',
      license = "BSD 3-Clause",
      install_requires = ["numpy>=1.7"],
      extras_requires = {
	  "ncio": "netCDF4>=1.0.6",
	  "pandas": "pandas>=0.8.0",
	  "plotting": ["pandas>=0.8.0"],
	  "interp2d": ["basemap>=1.06"],
	  }
      )
