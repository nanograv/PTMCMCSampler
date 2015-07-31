# -*- coding: utf-8 -*-

from __future__ import division

__all__ = ["DifferentialEvolutionProposal"]

import numpy as np

from .mh import MHProposal
from ..backends import DefaultBackend
from .. import dictutils as dt


class DifferentialEvolutionProposal(MHProposal):
    """
    Differential evolution proposal.


    :param pardict:
        Mapping dictionary for adaptive jump parameters.
        Format dict['parname'] = parameter index
    
    :param groups: (optional)
        Parameter ndices of separate jump groups.

    """

    def __init__(self, pardict, groups=None):

        self.pardict = pardict
        self.ndim = len(pardict)
        
        # get parameter groups
        if groups is None:
            self.groups = [np.arange(0, self.ndim)]
        else:
            self.groups = groups
        
        # DE buffer
        self.buffer = None

        proposal = self.get_DE_proposal
        name = 'DEProposal'

        super(DifferentialEvolutionProposal, self).__init__(proposal, name)
        
        
    def get_DE_proposal(self, coords, iteration, beta, pardict=None):
        """
        Differential evolution jump.
        
        :param coords: 
            The current set of parameter coordinates.

        :param iter: 
            Iteration of sampler

        :param beta:
            Inverse temperature of chain

        :param beta:
            Inverse temperature of chain

        :param pardict: (optional)
            Parameter dictionary from model class. Only necessary if using
            Trans-dimensional sampler.

        :returns: q: 
            New position in parameter space

        :returns: lqxy: 
            Logarithm of proposal density

        """

        # initial parameters
        q = coords.copy()
        lqxy = 0
        
        # check to see if buffer is empty
        if np.size(self.buffer) == 0:
            return q, lqxy

         # choose group
        jumpind = np.random.randint(0, len(self.groups))
        ndim = len(self.groups[jumpind])

        # adjust step size
        cd = 2.4 / np.sqrt(2*ndim)
        p = [0.1, 0.3, 0.55, 0.05]
        scales = [0.2, 0.5, 1, 10]
        scale = np.random.choice(scales, 1, p=p) * cd

        # adjust scale based on temperature
        scale *= np.sqrt(max(1/beta, 1e-2))
        
        indexmap = self.groups[jumpind]

        bufsize = self.buffer.shape[0]
        
        # draw a random integer from 0 - iter
        mm = np.random.randint(0, bufsize)
        nn = np.random.randint(0, bufsize)

        for ii in range(ndim):

            # jump size
            sigma = self.buffer[mm, self.groups[jumpind][ii]] - \
                    self.buffer[nn, self.groups[jumpind][ii]]

            # jump
            q[indexmap[ii]] += scale * sigma 

        return q, lqxy

    def setup(self, model, backend, iteration):
        """
        Read in buffer from backend to use for jumps.

        :param model:
            The starting :class:`Model`.
        
        :param backend:
            Instance of :class:`Backend` forom :class:`Sampler`.
        
        :param iteration:
            Iteration of MCMC sampler.
         """

        self.buffer = backend.buffer[:,np.array(self.pardict.values())]


