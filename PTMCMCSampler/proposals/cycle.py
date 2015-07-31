# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["JumpCycle"]

import numpy as np


class JumpCycle(object):
    """
    Class to create and store randomized jump schedules
    for use in the MCMC sampler.
    """

    def __init__(self):

        self.propCycle = []
        self.aux = []

    def add_proposal_to_cycle(self, proposal, weight):
        """
        Add jump proposal distributions to cycle with a given weight.

        :param proposal: 
            jump proposal object. Must have ``update``, ``auxiliary``
            and ``finalize`` methods.

        :param weight: 
            jump proposal weight in cycle

        """

        # get length of cycle so far
        length = len(self.propCycle)

        # check for 0 weight
        if weight == 0:
            print('ERROR: Can not have 0 weight in proposal cycle!')
            sys.exit()

        # add proposal to cycle
        for ii in range(length, length + weight):
            self.propCycle.append(proposal)

    def add_auxiliary_jump(self, func):
        """
        Add auxilary jump proposal distribution. This will be called after every
        standard jump proposal. Examples include cyclic boundary conditions and
        pulsar phase fixes

        :param func: 
            Auxiliary jump proposal function

        """

        # set auxilary jump
        self.aux.append(func)

    def finalize_prop_cycle(self):
        """
        Make sure all auxiliary proposals are populated in 
        individual proposal object.
        """

        for prop in self.propCycle:
            if self.aux != prop.aux:
                for aux in self.aux:
                    prop.add_auxiliary(aux)


    def draw_proposal(self):
        """Randomly draw proposal from cycle"""

        return self.propCycle[np.random.randint(0, len(self.propCycle))]




