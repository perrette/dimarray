from distutils.core import setup

with open('README') as file:
        long_description = file.read()

setup(name='dimarray',
      version='0.1',
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='array with labelled dimensions and axes, metadata and NaN handling',
      keywords=('labelled array','numpy','larry','pandas','iris'),
      packages = ['dimarray','dimarray.core','dimarray.geo','dimarray.io','dimarray.lib'],
      long_description=long_description,
      url='https://github.com/perrette/dimarray',
      license = "BSD 3-Clause",
      )
