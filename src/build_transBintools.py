# To build transBintools from .pyx source, run the following line:
# python build_transBintools.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize

setup(name='_transBintools', ext_modules=cythonize("_transBintools.pyx"))
