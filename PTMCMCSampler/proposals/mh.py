# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["MHProposal"]

import numpy as np


class MHProposal(object):
    """
    A general Metropolis-Hastings proposal class. Each update will call the 
    proposal, then an optional axiliary proposal to adjust the parameters.
    It then computes the hastings ratio and updates the Model class in place.
    It finishes with a user defined finalize function to do any clean up or 
    bookkeeping.

    :param proposal:
        The proposal function. It should take 4 arguments: the current set of
        parameters, the iteration number of the sampler, the inverse temperature
        of the chain (only used for parallel tempering), and the parameter index
        mapping dictionary.

    :param name:
        Name of proposal distribution.
    """

    def __init__(self, proposal_function, name):

        self.proposal = proposal_function
        self.name = name
        self.acceptance = False

        # auxiliary proposals
        self.aux = []

    def update(self, model, backend, iteration, beta=1):
        """
        Execute a single step starting from the given :class:`Model` and
        updating it in-place.

        :param model:
            The starting :class:`Model`.
        
        :param backend:
            Instance of :class:`Backend` forom :class:`Sampler`.
        
        :param iteration:
            Iteration of MCMC sampler.

        :param beta: (optional)
            Inverse temperature of parallel tempering chain.
            Defaults to 1

        :return model:
            The same model updated in-place.
        """

        # setup proposal
        self.setup(model, backend, iteration)

        # Compute the proposal.
        q, lqxy = self.proposal(model.coords, iteration,
                               beta, pardict=model.pardict)

        # auxiliary proposal before computing likelihood
        q, lqxy_aux = self.auxiliary(model.coords, q)
        lqxy += lqxy_aux
        
        # compute likelihood
        logprior, loglike, logpost = model.logpostfn(q, beta)

        # Hastings step
        model.acceptance = False
        self.acceptance = False
        H = np.exp(logpost - model.logpost + lqxy)
        if H > np.random.rand():
            model.acceptance = True
            self.acceptance = True
            model.update(q, loglike, logprior, beta)
        
        # finalize proposal
        self.finalize(model, backend, iteration)

        return model
    
    def setup(self, model, backend, iteration):
        """
        Do any proposal setup.

        :param model:
            The starting :class:`Model`.
        
        :param backend:
            Instance of :class:`Backend` forom :class:`Sampler`.
        
        :param iteration:
            Iteration of MCMC sampler.
         """
        pass

    def finalize(self, model, backend, iteration):
        """
        Do any proposal dependent cleaning or bookkeeping. Can be
        implemented with subclasses.

        :param model:
            The starting :class:`Model`.
        
        :param backend:
            Instance of :class:`Backend` forom :class:`Sampler`.
        
        :param iteration:
            Iteration of MCMC sampler.
         """
        pass

    def auxiliary(self, current, proposed):
        """
        Auxiliary jump to make adjustments to proposed parameters.

        :param current:
            The current paramters.

        :param proposed:
            The proposed parameters.

        :returns: 
            New parameter and any additional proposal density from
            this auxiliary function.
        """
        
        lqxy_aux = 0
        if len(self.aux) > 0:
            for aux in self.aux:
                proposed, lqxy = aux(current, proposed)
                lqxy_aux += lqxy

        return proposed, lqxy_aux

    
    def add_auxiliary(self, aux):
        """
        Add auxiliary function to the list.

        :param aux:
            Auxiliary proposal function to be called after every
            proposal. Takes 2 arguments, the current parameter vector
            and the proposed parameter vector.
        """

        self.aux.append(aux)
