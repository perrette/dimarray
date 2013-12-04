from distutils.core import setup

with open('README') as file:
        long_description = file.read()

setup(name='laxarray',
      version='1.2',
      py_modules=['laxarray'],
      description='Labelled Axis Array',
      long_description=long_description,
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      url='https://github.com/perrette/laxarray',
      license = "BSD 2-Clause"
      )
