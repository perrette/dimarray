#from distutils.core import setup
from setuptools import setup

with open('README.rst') as file:
        long_description = file.read()

setup(name='dimarray',
      version='0.1.7',
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='numpy array with labelled dimensions and axes, dimension, NaN handling and netCDF I/O',
      keywords=('labelled array','numpy','larry','pandas','iris'),
      packages = ['dimarray','dimarray.core','dimarray.geo','dimarray.io','dimarray.lib', 'dimarray.compat'],
      package_data = {
	  "dimarray": ['README.rst','dimarray.ipynb']
	  },
      long_description=long_description,
      url='https://github.com/perrette/dimarray',
      license = "BSD 3-Clause",
      install_requires = ["numpy>=1.7"],
      extras_require = {
	  "ncio": ["netCDF4>=1.0.6"],
	  "pandas": ["pandas>=0.8.0"],
	  "plotting": ["pandas>=0.8.0"],
	  }
      )
