# -*- coding: utf-8 -*-

from __future__ import division

__all__ = ["SingleAdaptiveGaussianProposal"]

import numpy as np

from .mh import MHProposal
from ..backends import DefaultBackend
from .. import dictutils as dt


class SingleAdaptiveGaussianProposal(MHProposal):
    """
    Single parameter adaptive metropolis from LALInference paper.

    :param sigma: 
        Initial jump sizes. Must be in same order as pardict
        values.

    :param pardict:
        Mapping dictionary for adaptive jump parameters.
        Format dict['parname'] = parameter index

    :param pmin:
        Minimum range of parameters. Must be in same
        order as pardict.
    
    :param pmax:
        Maximum range of parameters. Must be in same
        order as pardict.
    
    :param group: (optional)
        Parameter group indices. Must be in same order
        as pardict. Default is to use all parameters.

    :param adapt_iter: (optional)
        Number of iterations over which adaptation takes 
        place. Default is 100,000.

    :param target_acceptance: (optional)
        Target acceptance rate for individual proposals.

    """

    def __init__(self, sigma, pardict, pmin, pmax, group=None,
                adapt_iter=100000, target_acceptance=0.234):

        self.sigma = sigma
        self.pardict = pardict
        self.ndim = len(pardict)
        self.delta = pmax - pmin

        if group is None:
            self.group = np.arange(0, self.ndim)
        else:
            self.group = group

        self.adapt_iter = adapt_iter
        self.target_acceptance = target_acceptance

        self.active_parameter = self.group[0]

        proposal = self.get_proposal
        name = 'SingleAdaptiveGaussian'

        super(SingleAdaptiveGaussianProposal, self).__init__(proposal, name)


    def finalize(self, model, backend, iter):
        """
        Finalize method will update covariance matrix every
        cov_update steps.

        :param: 
            The starting :class:`Model`.
        
        :param backend:
            Instance of :class:`Backend` forom :class:`Sampler`.

        :param iter: 
            Iteration of sampler

        """

        if iter <= self.adapt_iter:
            sg = 10 * (iter+1)**(-1/5) - 1

            if model.acceptance == True:
                self.sigma[self.active_parameter] += \
                        sg * (1-self.target_acceptance) * \
                        self.delta[self.active_parameter] / 100

            else:
                self.sigma[self.active_parameter] -= \
                        sg * self.target_acceptance * \
                        self.delta[self.active_parameter] / 100



    def get_proposal(self, coords, iteration, beta, pardict=None):
        """
        Single Adaptive Gaussian Jump Proposal.

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

         # choose parameter
        jumppar = np.random.choice(self.pardict.keys(), 1)
        self.active_parameter = self.pardict[jumppar[0]]

        # get index mapping for shared parameters
        if pardict is not None:
            tmpdict = dt.get_dict_subset(self.pardict, [self.active_parameter])
            indexmap = dt.match_dict_values(pardict, tmpdict)
        else:
            indexmap = self.active_parameter
        
        q[indexmap] += np.random.randn() * self.sigma[self.active_parameter]

        return q, lqxy



