# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ =  ["DefaultBackend"]

import numpy as np


class DefaultBackend(object):
    """
    Default backend uses numpy arrays to store output as MCMC
    progresses.

    :param thin: (optional)
        Thinning factor for chain. Will update most arrays
        every `thin` steps. Default value is 10.

    :param bufsize: (optional)
        Buffer size for adaptive jumps. Default value is
        10000.
    """

    def __init__(self, thin=10, bufsize=10000):
        self.thin = thin
        self.bufsize = bufsize
        self.reset()

    def reset(self):
        """
        Clear the chain and reset it to its default state.
        """
        # Clear the chain dimensions.
        self.niter = 0
        self.size = 0
        self.ndim = None

        # Clear the chain wrappers.
        self._coords = None
        self._buffer = None
        self._logprior = None
        self._loglike = None
        self._logpost = None

        # Clear jump dictionary
        self._jumpdict = {}

    def check_dimensions(self, model, thin):
        """
        Check that model dimensions are set.

        :param model:
            The :class:`Model` class

        :param thin:
            Thin argument from the sampler
        """

        if self.ndim is None:
            self.ndim = model.ndim
        if self.thin != thin:
            self.thin = thin
        if self.ndim != model.ndim:
            raise ValueError("Dimension mismatch")

    def extend(self, n):
        """
        Extend output arrays by n.

        :param n:
            Amount by which to extend arrays.
        """

        d = self.ndim
        self.size = l = self.niter + n
        if self._coords is None:
            self._coords = np.empty((l // self.thin, d), dtype=np.float64)
            self._buffer = np.empty((self.bufsize, d), dtype=np.float64)
            self._logprior = np.empty(l // self.thin, dtype=np.float64)
            self._loglike = np.empty(l // self.thin, dtype=np.float64)
            self._logpost = np.empty(l // self.thin, dtype=np.float64)
            self._acceptance = 0
        else:
            self._coords = np.resize(self._coords, (l, d))
            self._logprior = np.resize(self._logprior, l)
            self._loglike = np.resize(self._loglike, l)
            self._logpost = np.resize(self._logpost, l)

    def update_model(self, model):
        """
        Update output arrays.

        :param model:
            The :class:`Model` class
        """

        i = self.niter
        if i % self.thin == 0:

            if i >= self.size:
                self.extend(i - self.size + 1)

            self._coords[i // self.thin] = model.coords
            self._logprior[i // self.thin] = model.logprior
            self._loglike[i // self.thin] = model.loglike
            self._logpost[i // self.thin] = model.logpost

        self._buffer[i % self.bufsize] = model.coords
        self._acceptance += model.acceptance
        self.niter += 1

    def update_proposal(self, proposal):
        """
        Update proposal statistics

        :param model:
            A proposal object
        """

        # update if not in dictionary
        if proposal.name not in self._jumpdict:
            self._jumpdict[proposal.name] = [0, 0]

        # update jump counter
        self._jumpdict[proposal.name][0] += 1

        # update acceptance counter
        self._jumpdict[proposal.name][1] += proposal.acceptance


    @property
    def coords(self):
        return self._coords[:(self.niter // self.thin)]

    @property
    def buffer(self):
        return self._buffer[:min(self.niter, self.bufsize)]

    @property
    def logprior(self):
        return self._logprior[:(self.niter // self.thin)]

    @property
    def loglike(self):
        return self._loglike[:(self.niter // self.thin)]

    @property
    def logpost(self):
        return self._logpost[:(self.niter // self.thin)]

    @property
    def acceptance(self):
        return self._acceptance

    @property
    def acceptance_fraction(self):
        return self.acceptance / self.niter

    @property
    def jumpdict(self):
        return self._jumpdict
