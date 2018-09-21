import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup, Extension

from topdot import __version__

ext_utils = Extension(
  'topdot.topdot',
  sources=['./topdot/topdot.pyx', './topdot/_topdot.cpp'],
  include_dirs=[numpy.get_include()],
  #libraries=[],
  extra_compile_args=['-std=c++0x', '-Os', '-fopenmp'],
  language='c++',
)

setup(
  name='topdot',
  version=__version__,
  setup_requires=[
    'setuptools>=18.0',
    'cython',
  ],
  packages=['topdot'],
  cmdclass={'build_ext': build_ext},
  ext_modules=cythonize([ext_utils]),
)
