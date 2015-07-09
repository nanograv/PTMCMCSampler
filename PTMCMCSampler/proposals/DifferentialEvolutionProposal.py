# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from ..backends import DefaultBackend
from .. import dictools as dt


class DifferentialEvolutionProposal(MHProposal):
    """
    Differential evolution proposal.


    :param pardict:
        Mapping dictionary for adaptive jump parameters.
        Format dict['parname'] = parameter index
    
    :param groups: (optional)
        Parameter ndices of separate jump groups.

    :param backend: (optional)
        Backend for storing data. Defaults to standard numpy
        array based backend. Other options include ['hdf5']

    """

    def __init__(self, pardict, groups=None, backend=None):

        self.pardict = pardict
        self.ndim = len(pardict)
        
        # get parameter groups
        if groups is None:
            self.groups = [np.arange(0, self.ndim)]
        else:
            self.groups = groups
        
        # determine backend
        if backend is None:
            self.backend = DefaultBackend()
        else:
            self.backend = backend

        proposal = get_DE_proposal
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

         # choose group
        jumpind = np.random.randint(0, len(self.groups))
        ndim = len(self.groups[jumpind])

        # adjust step size
        cd = 2.4 / np.sqrt(2*ndim)
        p = [0.1, 0.3, 0.55, 0.05]
        scales = [0.2, 0.5, 1, 10]
        scale = np.random.choice(scales, 1, p=p) * cd

        # adjust scale based on temperature
        scale *= np.sqrt(max(beta, 1e-2))
        
        # get index mapping for shared parameters
        if pardict is not None:
            tmpdict = dt.get_dict_subset(self.pardict, self.groups[jumpind])
            indexmap = dt.match_dict_values(pardict, tmpdict)
        else:
            indexmap = self.groups[jumpind]

        samples = self.backend.buffer[:,np.array(self.pardict.values())]
        samples = samples[np.flatnonzero(samples), :]
        bufsize = samples.shape[0]
        
        # draw a random integer from 0 - iter
        mm = np.random.randint(0, bufsize)
        nn = np.random.randint(0, bufsize)

        for ii in range(ndim):

            # jump size
            sigma = samples[mm, self.groups[jumpind][ii]] - \
                    samples[nn, self.groups[jumpind][ii]]

            # jump
            q[indexmap] += scale * sigma

        return q, lqxy


