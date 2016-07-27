from . import settings
from . import exceptions
from .enums import *
from .core import *

from . import _probes, signal, geometry, io, im, registration, has_cuda_gpu

probes = _probes.probes

__author__ = "UoB: Nicolas Budyn, Rhodri Bevan"
__credits__ = []
__license__ = "All rights reserved"
__copyright__ = "2016"

# Must respect PEP 440: https://www.python.org/dev/peps/pep-0440/
# Must be bumped at each release
__version__ = '0.4.dev1'
