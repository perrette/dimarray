from distutils.core import setup
setup(name='laxarray',
      version='1.0',
      py_module=['laxarray'],
      description='Labelled Axis Array',
      long_description="numpy's ndarrays like with a name per axis and nan treated as missing values",
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      url='https://github.com/perrette/laxarray',
      license = "BSD 2-Clause"
      )
