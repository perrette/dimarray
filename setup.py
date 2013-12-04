from distutils.core import setup

with open('README') as file:
        long_description = file.read()

setup(name='laxarray',
      version='1.2.1',
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='Labelled Axis Array',
      keywords=('labelled array','numpy','larry','pandas'),
      py_modules=['laxarray'],
      long_description=long_description,
      url='https://github.com/perrette/laxarray',
      license = "BSD 2-Clause",
      classifiers = [
	  "Development Status :: 3 - Alpha"
	  ]
      )
