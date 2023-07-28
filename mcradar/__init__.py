# -*- coding: utf-8 -*-

"""Top-level package for McRadar."""


__author__ = "José Dias Neto"
__email__ = "jdiasn@gmail.com"
__version__ = "0.0.1"


from .utilities import *
from .settings import loadSettings
from .tableOperator import getMcSnowTable

from .radarOperator.spectraOperator import getMultFrecSpec
from .radarOperator.spectraOperator import convoluteSpec
from .radarOperator.zeOperator import calcParticleZe
from .radarOperator.kdpOperator import getIntKdp

from .fullRadarOperator import fullRadar

from .fullRadarOperator import singleParticleTrajectories
from .fullRadarOperator import singleParticleScat
