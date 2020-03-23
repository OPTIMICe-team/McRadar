# -*- coding: utf-8 -*-


__author__ ='José Dias Neto'
__email__ ='jdiasn@gmail.com'
__version__ = '0.0'


from .utilities import *
from .settings import loadSettings
from .tableOperator import getMcSnowTable

from .radarOperator.spectraOperator import getMultFrecSpec
from .radarOperator.zeOperator import calcParticleZe
from .radarOperator.kdpOperator import calcParticleKDP, getIntKdp

from .fullRadarOperator import fullRadar
