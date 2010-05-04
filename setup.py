from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import os
from numpy.distutils.misc_util import *
numpyincl = get_numpy_include_dirs()

psslda_module = Extension("pSSLDA", ["pSSLDA.pyx"],
                          include_dirs = [os.getcwd()] + numpyincl)

flda_module = Extension("FastLDA",
                        sources = ["FastLDA.c"],
                        include_dirs = [os.getcwd()] + numpyincl,
                        library_dirs = [],
                        libraries = [],
                        extra_compile_args = ['-O3','-Wall'],
                        extra_link_args = [])

py_mods = []

setup(name = 'pSSLDA',
      description = 'Parallel Semi-Supervised LDA',
      version = '0.0.0',
      author = 'David Andrzejewski',
      author_email = 'andrzeje@cs.wisc.edu',
      license = 'GNU General Public License (Version 3 or later)',
      url = 'http://pages.cs.wisc.edu/~andrzeje',
      cmdclass = {'build_ext': build_ext},
      ext_modules = [psslda_module, flda_module],
      py_modules = py_mods)
