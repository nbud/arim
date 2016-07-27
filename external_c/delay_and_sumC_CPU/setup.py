from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


ext_module = Extension(
    "delay_and_sumC_CPU",
    ["delay_and_sumC_CPU.pyx", "delay_and_sum_nearest_DP_CPU.c", "delay_and_sum_linear_DP_CPU.c", "delay_and_sum_linear_SP_CPU.c", "delay_and_sum_nearest_SP_CPU.c"],
    extra_compile_args=['/Ox','/openmp','/arch:AVX2'],
    extra_link_args=[''],
    language='c++'
)

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

setup(
    name='delay_and_sumC_CPU',
    version='0.2',
    description='Delay And Sum Algorithms for CPU (VARIANTS FROM SP/DP/NEAREST/LINEAR)',
    author='Rhodri Bevan, UoB',
    cmdclass = {'build_ext': build_ext},    
    ext_modules = [ext_module],
    include_dirs=[numpy_include]

)

