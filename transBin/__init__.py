"""transBin
Documentation: https://github.com/RasmussenLab/transBin/

transBin contains the following modules:
transBin.transBintools
transBin.parsecontigs
transBin.parsebam
transBin.encode
transBin.cluster
transBin.benchmark

General workflow:
1) Filter contigs by size using transBin.transBintools.filtercontigs
2) Map reads to contigs to obtain BAM file
3) Calculate TNF of contigs using transBin.parsecontigs
4) Create RPKM table from BAM files using transBin.parsebam
5) Train autoencoder using transBin.encode
6) Cluster latent representation using transBin.cluster
7) Split bins using transBin.transBintools
"""

__authors__ = 'Jakob Nybo Nissen', 'Simon Rasmussen'
__licence__ = 'MIT'
__version__ = (3, 0, 9)

import sys as _sys
if _sys.version_info[:2] < (3, 5):
    raise ImportError('Python version must be >= 3.5')

from . import transBintools
from . import parsebam
from . import parsecontigs
from . import cluster
from . import benchmark
from . import transformer
