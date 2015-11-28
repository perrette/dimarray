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
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys,subprocess
        # errno = subprocess.call([sys.executable, 'runtest.py'])
        errno = subprocess.call(['py.test'])
        raise SystemExit(errno)


cmdclass.update({'test':MyTests})

setup(name='dimarray',
      version=versioneer.get_version(),
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='numpy array with labelled dimensions and axes, dimension, NaN handling and netCDF I/O',
      keywords=('labelled array','numpy','larry','pandas','iris'),
      packages = ['dimarray','dimarray.core','dimarray.geo','dimarray.io','dimarray.lib', 'dimarray.compat','dimarray.datasets','dimarray.convert'],
      package_data={'dimarray.datasets': ['data/*']},

      # long_description=long_description,
      url='https://github.com/perrette/dimarray',
      license = "BSD 3-Clause",
      requires = ["numpy(>=1.7)"],
      cmdclass = cmdclass,
      )

