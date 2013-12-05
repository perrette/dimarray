from distutils.core import setup

with open('README') as file:
        long_description = file.read()

setup(name='dimarray',
      version='1.2.1',
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='Labelled Axis Array',
      keywords=('labelled array','numpy','larry','pandas','iris'),
      py_modules=['dimarray'],
      long_description=long_description,
      url='https://github.com/perrette/dimarray',
      license = "BSD 2-Clause",
      )
