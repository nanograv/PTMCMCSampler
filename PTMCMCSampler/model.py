# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

class Model(object):
    """
    A :class:`Model` is a containier for likelihood functions, priors,
    coordinate positions etc. 

    :param coords:
        Initial set of coordinates in parameter space

    :param pardict:
        Mapping dictionary for parameters.
        Format: dict['parname'] = parameter index

    :param loglikefn: 
        Log likelihood function.

    :param logpriorfn:
        Log prior function

    :param loglargs: (optional)
        Additional non-keyword arguments to log likelihood function.

    :param loglkwargs: (optional)
        Additional keyword arguments to log likelihood function.

    :param logpargs: (optional)
        Additional non-keyword arguments to log prior function.

    :param logpkwargs: (optional)
        Additional nkeyword arguments to log prior function.

    """
    def __init__(self, coords, pardict, loglikefn, logpriorfn, loglargs=[],
                 loglkwargs={}, logpargs=[], logpkwargs={}):

        # wrap log likelihood and prior functions
        self._loglikefn = _function_wrapper(loglikefn, loglargs, loglkwargs)
        self._logpriorfn = _function_wrapper(logpriorfn, logpargs, logpkwargs)

        # get coordinates and number of parameters
        self._coords = coords
        self._pardict = pardict

        if len(self.coords) != len(self.pardict):
            raise ValueError('Parameter dictionary and coords do' 
                             ' not have same length')
        self.ndim = len(self.pardict)


        # Save the initial prior and likelihood values.
        self._logprior, self._loglike, self._logpost = self.logpostfn(self.coords) 
        self.acceptance = True

        # Check the initial probabilities.
        if not (np.all(np.isfinite(self._logprior))
                and np.all(np.isfinite(self._loglike))):
            raise ValueError("Invalid (un-allowed) initial coordinates")

    def logpostfn(self, coords, beta=1):
        """
        This method computes the natural logarithm of the posterior
        probability up to a constant. 
        
        :param coords:
            The coordinates where the probability should be evaluated.

        :param beta: (optional)
            Inverse temperature in PT scheme. Default is 1.
        """

        lp = self._logpriorfn(coords)
        if not np.isfinite(lp):
            return -np.inf, -np.inf, -np.inf

        ll = self._loglikefn(coords,)
        if not np.isfinite(ll):
            return lp, -np.inf, -np.inf

        return lp, ll, lp + ll*beta


    def update(self, coords, loglike, logprior, beta=1):
        """
        Update the coordinates and probability containers given the
        current set of parameters. Proposals should call this after 
        proposing.

        :param coords:
            The new parameter coordinates.

        :param loglike:
            The new log likelihood.

        :param logprior:
            The new logprior
        
        :param beta: (optional)
            Inverse temperature in PT scheme. Default is 1.

        """

        self._coords, self._loglike, self._logprior = coords, loglike, logprior
        self._logpost = beta*loglike + logprior

        # Check the probabilities and make sure that no invalid samples were
        # accepted.
        if not (np.isfinite(self._coords)
                and np.isfinite(self._logprior)
                and np.isfinite(self._loglike)):
            raise RuntimeError("An invalid proposal was accepted")


    @property
    def coords(self):
        """The coordinate vector."""
        return self._coords
    
    @property
    def pardict(self):
        """The Parameter index dictionary"""
        return self._pardict

    @property
    def logprior(self):
        """The log-prior."""
        return self._logprior

    @property
    def loglike(self):
        """The log-likelihood up to a constant."""
        return self._loglike

    @property
    def logpost(self):
        """The log-probabilitie up to a constant."""
        return self._logpost


class _function_wrapper(object):

"""
This is a hack to make the likelihood function pickleable when ``args``
or ``kwargs`` are also included.

"""

def __init__(self, f, args, kwargs):
    self.f = f
    self.args = args
    self.kwargs = kwargs

def __call__(self, x):
    return self.f(x, *self.args, **self.kwargs)
