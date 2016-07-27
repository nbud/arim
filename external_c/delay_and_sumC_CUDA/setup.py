from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import os
from os.path import join as pjoin
#import numpy

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDA_PATH env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # On Windows look for 'nvcc.exe' rather than 'nvcc'
    if os.name == 'nt':
        nvcc_exe = 'nvcc.exe'
        lib64_part = 'lib/x64'
    else:
        nvcc_exe = 'nvcc'
        lib64_part = 'lib64'

    # first check if the CUDA_PATH env variable is in use
    if 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
        nvcc = pjoin(home, 'bin', nvcc_exe)
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path(nvcc_exe, os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDA_PATH')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, lib64_part)}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()

#try:
#    numpy_include = numpy.get_include()
#except AttributeError:
#    numpy_include = numpy.get_numpy_include()

ext_modules=[
    Extension("delay_and_sumC_CUDA",
              sources=["delay_and_sumC_CUDA.pyx"],
              library_dirs=[CUDA['lib64'], '.'],
              libraries=['delay_and_sumC_CUDA', 'cudart'],
              language='c++',
    )
]


setup(
    name = "delay_and_sumC_CUDA",
    version='0.2',
    description='Delay And Sum Algorithms for CUDA GPU (VARIANTS FROM SP/DP/NEAREST/LINEAR)',
    author='Rhodri Bevan, UoB',
    ext_modules = cythonize(ext_modules),
)
