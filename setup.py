from setuptools import setup, find_packages
import os, sys
import re

import warnings
import versioneer
cmdclass = versioneer.get_cmdclass()

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(name='dimarray',
      version=versioneer.get_version(),
      author='Mahe Perrette',
      author_email='mahe.perrette@gmail.com',
      description='numpy array with labelled dimensions and axes, dimension, NaN handling and netCDF I/O',
      keywords=('labelled array','numpy','larry','pandas'),
      packages = find_packages(),
      package_data={'dimarray.datasets': ['data/*']},
      long_description=long_description,
      long_description_content_type="text/x-rst",
      url='https://github.com/perrette/dimarray',
      license = "BSD 3-Clause",
      install_requires = ["numpy(>=1.7)","future"],
      cmdclass = cmdclass,
      )

