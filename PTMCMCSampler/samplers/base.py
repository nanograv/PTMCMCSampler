# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Sampler"]

import numpy as np
import time
import sys

from ..backends import DefaultBackend


class Sampler(object):
    """
    A base sampler object.

    :param schedule:
        Initialized JumpCycle object containing proposal schedule.

    :param backend: (optional)
        Backend for storing MCMC variables. Uses DefaultBackend
        by default. Other optoins include ['ascii', 'hdf5']
    """

    def __init__(self, schedule, backend=None):

        # check length of jump cycle
        if len(schedule.propCycle) == 0:
            raise ValueError('proposal cycle is empty!')

        self.schedule = schedule

        # Set up the backend.
        if backend is None:
            self.backend = DefaultBackend()
        else:
            self.backend = backend

        # Set the chain to the original untouched state.
        self.reset()

    def reset(self):
        """
        Clear the chain and reset it to its default state.
        """
        self.backend.reset()


    def sample(self, model, niter=1000000, store=True, thin=10):
        """
        Starting from a given model, sample for a given number of iterations.

        :param model:
            The starting :class:`Model`.

        :param niter:
            The number of steps to run. Default is 1000000.

        :param store: (optional)
            If ``True``, save the chain using the backend. If ``False``,
            reset the backend and don't store anything. Default is ``True``.

        :param thin: (optional)
            Only store every ``thin`` step.
        """

        # Check that the thin keyword is reasonable.
        thin = int(thin)
        assert thin > 0, "Invalid thinning argument"

        # Check the model dimensions.
        if store:
            self.backend.check_dimensions(model, thin)
        else:
            self.backend.reset()

        # Extend the chain to the right length.
        if store:
            if niter is None:
                self.backend.extend(0)
            else:
                self.backend.extend(niter // thin)

        # Start the sampler
        iter = 0
        self.tstart = time.time()
        while True:
            
            # draw proposal
            p = self.schedule.draw_proposal()

            # update model
            model = p.update(model, self.backend, iter)

            # Store this update if required
            if store:
                self.backend.update_model(model)
                self.backend.update_proposal(p)

            # Finish the chain if the total number of steps was reached.
            iter += 1

            # outpt progress
            if iter % thin == 0:
                self.progress(iter, niter)

            if iter >= niter:
                return


    def progress(self, iter, niter):
        """
        Print chain progress along with acceptance rate to the screen.

        :param iter:
            Iteration of the MCMC

        :param niter:
            Total number of MCMC samples

        """

        sys.stdout.write('\r')
        sys.stdout.write('Finished %2.2f percent in %f s Acceptance rate = %g'
                         % (iter / niter * 100,
                            time.time() - self.tstart,
                            self.acceptance_fraction))
        sys.stdout.flush()


    @property
    def coords(self):
        return self.backend.coords

    @property
    def logprior(self):
        return self.backend.logprior

    @property
    def loglike(self):
        return self.backend.loglike

    @property
    def logpost(self):
        return self.backend.logpost

    @property
    def acceptance_fraction(self):
        return self.backend.acceptance_fraction

