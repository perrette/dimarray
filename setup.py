#!/usr/bin/env python
"""
The version tracking part of this file comes from pandas' setup.py 
which is under the BSD license
"""
#from distutils.core import setup
import os, sys
import re
from setuptools import setup
from setuptools.command.test import test as TestCommand 
#import dimarray  # just checking

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


with open('README.rst') as file:
    long_description = file.read()

#
# Track version after pandas' setup.py
#
MAJOR = 0
MINOR = 1
MICRO = 8
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git','git.cmd']:
        try:
            pipe = subprocess.Popen([cmd, "describe", "--always", "--match", "v[0-9]*"],
                                stdout=subprocess.PIPE)
            (so,serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if os.path.exists('dimarray/version.py'):
            warnings.warn("WARNING: Couldn't get git revision, using existing dimarray/version.py")
            write_version = False
        else:
            warnings.warn("WARNING: Couldn't get git revision, using generic version string")
    else:
      # have git, in git dir, but may have used a shallow clone (travis does this)
      rev = so.strip()
      # makes distutils blow up on Python 2.7
      if sys.version_info[0] >= 3:
          rev = rev.decode('ascii')

      if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}",rev):
          # partial clone, manually construct version string
          # this is the format before we started using git-describe
          # to get an ordering on dev version strings.
          rev ="v%s.dev-%s" % (VERSION, rev)

      # Strip leading v from tags format "vx.y.z" to get th version string
      FULLVERSION = rev.lstrip('v')

else:
    FULLVERSION += QUALIFIER

#
#
#
setup(name='dimarray',
      version=FULLVERSION,
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='numpy array with labelled dimensions and axes, dimension, NaN handling and netCDF I/O',
      keywords=('labelled array','numpy','larry','pandas','iris'),
      packages = ['dimarray','dimarray.core','dimarray.geo','dimarray.io','dimarray.lib', 'dimarray.compat','dimarray.datasets'],
      package_data = {
	  "dimarray": ['README.rst','dimarray.ipynb']
	  },
      long_description=long_description,
      url='https://github.com/perrette/dimarray',
      license = "BSD 3-Clause",
      install_requires = ["setuptools", "numpy>=1.7"],
      tests_require = ["pytest"],
      extras_require = {
	  "ncio": ["netCDF4>=1.0.6"],
	  "pandas": ["pandas>=0.11.0"],
	  "plotting": ["matplotlib>=1.1", "pandas>=0.11.0"],
	  },
      cmdclass = {'test':MyTests},
      )

def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'dimarray', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

# Write version.py to dimarray
if write_version:
    write_version_py()
