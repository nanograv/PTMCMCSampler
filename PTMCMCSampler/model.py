# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Model"]

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

    :param indicator:
        Boolean indicator array determining which parameters to include
        in likelihood function. Used for RJMCMC.

    :param loglargs: (optional)
        Additional non-keyword arguments to log likelihood function.

    :param loglkwargs: (optional)
        Additional keyword arguments to log likelihood function.

    :param logpargs: (optional)
        Additional non-keyword arguments to log prior function.

    :param logpkwargs: (optional)
        Additional nkeyword arguments to log prior function.

    """
    def __init__(self, coords, pardict, loglikefn, logpriorfn, indicator=None,
                 loglargs=[], loglkwargs={}, logpargs=[], logpkwargs={},
                 beta=1):

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
        
        # boolean indicator array
        if indicator is None:
            self.indicator = np.array([1]*self.ndim, dtype=bool)
        else:
            self.indicator = self.indicator

        # Save the initial prior and likelihood values.
        self._logprior, self._loglike, self._logpost = self.logpostfn(self.coords, beta) 
        self.acceptance = True

        # Check the initial probabilities.
        if not (np.all(np.isfinite(self._logprior))
                and np.all(np.isfinite(self._loglike))):
            raise ValueError("Invalid (un-allowed) initial coordinates")
    
    def logpriorfn(self, coords, **kwargs):
        """
        This method computes the natural logarithm of the prior
        probability up to a constant. 
        
        :param coords:
            The coordinates where the probability should be evaluated.

        :param kwargs: (optional)
            any additional keyword arguments to be passed to the
            logprior function.
        """

        lp = self._logpriorfn(coords, **kwargs)
        return lp
    
    def loglikefn(self, coords, **kwargs):
        """
        This method computes the natural logarithm of the likelihood
        probability up to a constant. 
        
        :param coords:
            The coordinates where the probability should be evaluated.

        :param kwargs: (optional)
            any additional keyword arguments to be passed to the
            loglike function.
        """

        ll = self._loglikefn(coords, **kwargs)
        return ll

    def logpostfn(self, coords, beta=1, **kwargs):
        """
        This method computes the natural logarithm of the posterior
        probability up to a constant. 
        
        :param coords:
            The coordinates where the probability should be evaluated.

        :param beta: (optional)
            Inverse temperature in PT scheme. Default is 1.

        :param kwargs: (optional)
            any additional keyword arguments to be passed to loglike
            and logprior function.
        """
        lp = self._logpriorfn(coords, **kwargs)
        if not np.isfinite(lp):
            return -np.inf, -np.inf, -np.inf

        ll = self._loglikefn(coords, **kwargs)
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
        if not (np.isfinite(np.all(self._coords))
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

    def __call__(self, x, **kwargs):
        kwargs1 = self.kwargs.copy()
        kwargs1.update(kwargs)
        return self.f(x, *self.args, **kwargs1)
