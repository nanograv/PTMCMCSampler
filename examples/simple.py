#!/usr/bin/env python
"""
Sample code for sampling a multivariate Gaussian using base sampler.
"""

from __future__ import print_function

import PTMCMCSampler as ptmcmc
import PTMCMCSampler.proposals as jumps

import numpy as np
import matplotlib.pyplot as plt


# Define log- likelihood function and prior
def logpriorfn(x):
    return 0.0

def loglikefn(x):
    return -0.5 * np.sum(x ** 2)

# Use a 5 dimensional gaussian
ndim = 5

pmin = np.ones(ndim) * -5
pmax = np.ones(ndim) * 5

# initial coordinates
p0 = np.random.randn(ndim)

# initial jump covariance matrix
cov0 = np.diag([0.1**2]*ndim)

# generic parameter dictionary
pardict = {}
for ii in range(ndim):
    key = 'par_{0}'.format(ii)
    pardict[key] = ii

# initialize the Model
model = ptmcmc.Model(p0, pardict, loglikefn, logpriorfn)

# setup jump cycle
single_jump = jumps.SingleAdaptiveGaussianProposal(
    np.sqrt(np.diag(cov0)), pardict, pmin, pmax)
am_jump = jumps.AdaptiveMetropolisProposal(cov0, pardict, type='AM')
scam_jump = jumps.AdaptiveMetropolisProposal(cov0, pardict, type='SCAM')
de_jump = jumps.DifferentialEvolutionProposal(pardict)

# add to jump cycle
cycle = ptmcmc.JumpCycle()
cycle.add_proposal_to_cycle(am_jump, 10)
#cycle.add_proposal_to_cycle(scam_jump, 20)
#cycle.add_proposal_to_cycle(single_jump, 20)
#cycle.add_proposal_to_cycle(de_jump, 20)
cycle.finalize_prop_cycle()

# initialize sampler
sampler = ptmcmc.Sampler(cycle)

# sample for 100000 samples
N = 100000
sampler.sample(model, niter=N, thin=1)

# plot samples
plt.figure()
plt.plot(sampler.logpost)
plt.show()

for ii in range(5):
    plt.subplot(3, 2, ii+1)
    plt.plot(sampler.coords[:,ii])

plt.show()


