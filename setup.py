#!/usr/bin/env python2.7
import os, sys
import re
from distutils.core import setup, Command as TestCommand
import warnings
import versioneer
cmdclass = versioneer.get_cmdclass()

class MyTests(TestCommand):
    """ from http://pytest.org/latest/goodpractises.html
    """
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        #self.pytest_args = ['--doctest-modules','--doctest-glob="*rst"']
        self.pytest_args = []

        # remove pyc files before testing ?
        # find . -name "*.pyc" -exec rm {} \;


    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


cmdclass.update({'test':MyTests})

# get netcdf datafiles
datafiles = [(root, [os.path.join(root, f) for f in files])
            for root, dirs, files in os.walk('dimarray/datasets/data')]

#
#
#
setup(name='dimarray',
      version=versioneer.get_version(),
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='numpy array with labelled dimensions and axes, dimension, NaN handling and netCDF I/O',
      keywords=('labelled array','numpy','larry','pandas','iris'),
      packages = ['dimarray','dimarray.core','dimarray.geo','dimarray.io','dimarray.lib', 'dimarray.compat','dimarray.datasets','dimarray.convert'],
      data_files = datafiles,
      # long_description=long_description,
      url='https://github.com/perrette/dimarray',
      license = "BSD 3-Clause",
      install_requires = ["numpy>=1.7"],
      tests_require = ["pytest"],
      extras_require = {
          "ncio": ["netCDF4>=1.0.6"],
          "pandas": ["pandas>=0.11.0"],
          "plotting": ["matplotlib>=1.1"],
          },
      cmdclass = cmdclass,
      )

