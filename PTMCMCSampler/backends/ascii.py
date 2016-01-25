# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ =  ["AsciiBackend"]

import numpy as np
import os
from .default import DefaultBackend


class AsciiBackend(DefaultBackend):
    """
    Ascii backend is derived from :class: `DefaultBackend` and
    stores values in ascii format.

    :param outdir:
        Full path to output directory where ascii files will be
        stored.

    :param thin: (optional)
        Thinning factor for chain. Will update most arrays
        every `thin` steps. Default value is 10.

    :param bufsize: (optional)
        Buffer size for adaptive jumps. Default value is
        10000.

    :param isave: (optional)
        Number of samples before file saves. Default value
        is 1000

    .. note:: For now this class stil keeps numpy arrays and only
              saves to ascii files. In the future it will do everything
              through ascii files.
    """

    def __init__(self, outdir, thin=10, bufsize=10000, isave=1000):

        self._outdir = outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass

        super(AsciiBackend, self).__init__(thin, bufsize)
        self.isave = isave

    def reset(self):
        """
        Clear the chain and reset it to its default state.
        """
        self.initialized = False
        super(AsciiBackend, self).reset()

    def extend(self, n):
        """
        Extend output arrays by n. Also initialize output ascii
        files if not initialized already.

        :param n:
            Amount by which to extend arrays.
        """
        super(AsciiBackend, self).extend(n)

        # setup files if not initialized
        if not self.initialized:
            self.chainfile = open(self._outdir + '/chain.txt', 'w')
            self.probfile = open(self._outdir + '/prob.txt', 'w')
            self.indfile = open(self._outdir + '/indicator.txt', 'w')

            self.chainfile.close()
            self.probfile.close()
            self.indfile.close()

    def update_model(self, model):
        """
        Update output arrays.

        :param model:
            The :class:`Model` class
        """
        super(AsciiBackend, self).update_model(model)

        i = self.niter
        if i % self.isave == 0 and i > 1:

            # open files for appending
            self.chainfile = open(self._outdir + '/chain.txt', 'a+')
            self.probfile = open(self._outdir + '/prob.txt', 'a+')
            self.indfile = open(self._outdir + '/indicator.txt', 'a+')
            
            for jj in range(i-self.isave, i, self.thin):
                ind = jj // self.thin
                self.chainfile.write('\t'.join(['%22.22f' % (self._coords[ind, kk]) 
                                                for kk in range(self.ndim)]))
                self.chainfile.write('\n')
                self.probfile.write('%22.22f %22.22f %22.22f\n' % (self._logprior[ind], 
                                                                   self._loglike[ind],
                                                                   self._logpost[ind]))
                self.indfile.write('\t'.join(['%d' % (self._indicator[jj, kk]) 
                                              for kk in range(self.ndim)])) 
                self.indfile.write('\n')

            self.chainfile.close()
            self.probfile.close()
            self.indfile.close()

            




