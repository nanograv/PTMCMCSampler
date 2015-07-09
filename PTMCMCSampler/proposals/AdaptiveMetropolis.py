# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from ..backends import DefaultBackend
from .. import dictools as dt


class AdaptiveMetropolisProposal(MHProposal):
    """
    Adaptive metropolis proposal.

    :param cov: 
        Initial covariance matrix.

    :param pardict:
        Mapping dictionary for adaptive jump parameters.
        Format dict['parname'] = parameter index
    
    :param groups: (optional)
        Parameter ndices of separate jump groups.

    :param cov_update: (optional)
        Number of iterations before covariance matrix
        update via recursive relation. Default value is
        1000

    :param backend: (optional)
        Backend for storing data. Defaults to standard numpy
        array based backend. Other options include ['hdf5']

    :param type: (optional)
        Type of adaptive metropolis jump ['AM', 'SCAM'].
        AM performs a jump in all parameters where SCAM
        only jumps along one eigenvector of the covariance
        at a time.

    """

    def __init__(self, cov, pardict, groups=None, cov_update=1000,
                 backend=None, type='AM'):

        self.cov = cov
        self.cov_update = cov_update
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

        # set up covariance groups
        self.U = [[]] * len(self.groups)
        self.S = [[]] * len(self.groups)

        # do svd on parameter groups
        for ct, group in enumerate(self.groups):
            covgroup = np.zeros((len(group), len(group)))
            for ii in range(len(group)):
                for jj in range(len(group)):
                    covgroup[ii, jj] = self.cov[group[ii], group[jj]]

            self.U[ct], self.S[ct], v = np.linalg.svd(covgroup)
        
        # auxiliary parameters for recursive covariance updates
        self.M2 = np.zeros((ndim, ndim))
        self.mu = np.zeros(ndim)

        # set up proposal
        if type == 'AM':
            proposal = self.get_AM_proposal
            name = 'AMProposal'
        elif type == 'SCAM':
            proposal = self.get_SCAM_proposal
            name = 'SCAMProposal'
        else:
            raise ValueError('Proposal type not recognized.')

        super(AdaptiveMetropolisProposal, self).__init__(proposal, name)


    def finalize(self, model, iter):
        """
        Finalize method will update covariance matrix every
        cov_update steps.

        :param: 
            The starting :class:`Model`.

        :param iter: 
            Iteration of sampler

        """

        if iter !=0 and iter % self.cov_update == 0:
            self._update_covariance(iter)


    def _update_covariance(self, iter):
        """
        Function to recursively update sample covariance matrix.

        :param iter: 
            Iteration of the Sampler.        

        """

        start = iter - self.cov_update
        ndim = self.ndim

        # get past cov_update samples
        samples = self.backend.buffer[-self.cov_update:,
                                      np.array(self.pardict.values())]

        if start == 0:
            self.M2 = np.zeros((ndim, ndim))
            self.mu = np.zeros(ndim)

        for ii in range(self.cov_update):
            start += 1

            diff = samples[ii,:] - self.mu
            self.mu += diff / start

            self.M2 += np.outer(diff, diff)
            
        self.cov = self.M2 / (start - 1)

        # do svd on parameter groups
        for ct, group in enumerate(self.groups):
            covgroup = np.zeros((len(group), len(group)))
            for ii in range(len(group)):
                for jj in range(len(group)):
                    covgroup[ii, jj] = self.cov[group[ii], group[jj]]
            
            # SVD for group covariances
            self.U[ct], self.S[ct], v = np.linalg.svd(covgroup)


    def get_SCAM_proposal(self, coords, iteration, beta, pardict=None):
        """
        Single Component Adaptive Jump Proposal.  It will occasionally 
        use different jump sizes to ensure proper mixing.

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
        cd = 2.4 / np.sqrt(2)
        p = [0.1, 0.3, 0.55, 0.05]
        scales = [0.2, 0.5, 1, 10]
        scale = np.random.choice(scales, 1, p=p) * cd

        # adjust scale based on temperature
        scale *= np.sqrt(max(beta, 1e-2))

        # make correlated componentwise adaptive jump
        ind = np.random.randint(0, ndim, 1)

        if pardict is not None:
            tmpdict = dt.get_dict_subset(self.pardict, self.groups[jumpind])
            indexmap = dt.match_dict_values(pardict, tmpdict)
        else:
            indexmap = self.groups[jumpind]

        q[indexmap] += np.random.randn() * scale * \
                np.sqrt(self.S[jumpind][ind]) * \
                self.U[jumpind][:, ind].flatten()

        return q, lqxy

    def get_AM_proposal(self, coords, iteration, beta, pardict=None):
        """
        Adaptive Metropolis Jump Proposal.  It will occasionally 
        use different jump sizes to ensure proper mixing.
        
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

        # "cholesky" decomposition of covariance matrix
        L = scale * np.dot(self.U[jumpind], np.sqrt(self.S[jumpind]))

        # get index mapping for shared parameters
        if pardict is not None:
            tmpdict = dt.get_dict_subset(self.pardict, self.groups[jumpind])
            indexmap = dt.match_dict_values(pardict, tmpdict)
        else:
            indexmap = self.groups[jumpind]

        # multivariate quassian jump
        q[indexmap] += np.dot(L, np.random.randn(ndim))

        return q, lqxy


