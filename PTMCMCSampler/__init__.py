from __future__ import print_function, division

__version__ = 2015.03

__all__ = ["proposals", "backends", "Sampler", "Model",
           "JumpCycle"]

from . import backends, proposals, dictutils
from .proposals import JumpCycle
from .samplers import Sampler
from .model import Model

def test():
    # Run some tests here
    print("{0} tests have passed".format(0))
