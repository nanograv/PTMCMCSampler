"""
A Python script that repackages PTMCMCSampler's "simple" example to run from the command line using
mpirun.  https://github.com/jellis18/PTMCMCSampler/blob/master/examples/simple.ipynb
------------------
Runtime environment:
This script requires a Python runtime environment with installed dependencies, and also a working
MPI installation.

------------------
Sample commands:

Show help for the script:
mpi4py -m gaussian_mpi_example -h

Testing the script (one chain):
python -m gaussian_mpi_example [output_dir]

Basic use (multiple chains):
mpirun -np [n_chains] python -m gaussian_mpi_example [output_dir]

Running 2 chains with repeatable results:
mpirun -np 2 python -m gaussian_mpi_example --seeds 10 11 /code/results/2-chains

Result files are written to the configured output directory, overwriting any previous results
with the same names.  Clients are responsible to clean up result files generated by the script
(from the PTMCMCSampler library).
See also supporting library PTMCMCSampler https://github.com/jellis18/PTMCMCSampler
"""
import argparse
import logging
import random

import numpy as np
from mpi4py import MPI

from PTMCMCSampler import PTMCMCSampler

logger = logging.getLogger(__name__)


class GaussianLikelihood:
    def __init__(self, ndim=2, pmin=-10, pmax=10):

        self.a = np.ones(ndim) * pmin
        self.b = np.ones(ndim) * pmax

        # get means
        self.mu = np.random.uniform(pmin, pmax, ndim)

        # ... and a positive definite, non-trivial covariance matrix.
        cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
        cov = np.triu(cov)
        cov += cov.T - np.diag(cov.diagonal())
        self.cov = np.dot(cov, cov)

        # Invert the covariance matrix first.
        self.icov = np.linalg.inv(self.cov)

    def lnlikefn(self, x):
        diff = x - self.mu
        return -np.dot(diff, np.dot(self.icov, diff)) / 2.0

    def lnpriorfn(self, x):

        if np.all(self.a <= x) and np.all(self.b >= x):
            return 0.0
        else:
            return -np.inf


class UniformJump:
    def __init__(self, pmin, pmax):
        """Draw random parameters from pmin, pmax"""
        self.pmin = pmin
        self.pmax = pmax

    def jump(self, x, it, beta):
        """
        Function prototype must read in parameter vector x,
        sampler iteration number it, and inverse temperature beta
        """

        # log of forward-backward jump probability
        lqxy = 0

        # uniformly drawm parameters
        q = np.random.uniform(self.pmin, self.pmax, len(x))

        return q, lqxy


def parallel_tempering_opt(output_dir, verbose=False):
    """
    Runs the parallel tempering example based on code from PTMCMCSampler's simple.ipynb.
    Note this function purposefully omits a random seed parameter.  For repeatable results when
    calling it directly, call np.random.seed() in client code (and only once per process,
    since otherwise it may cause random numbers to be repeated).
    :param output_dir: the output directory for results
    """

    # Setup Gaussian model class
    ndim = 20
    pmin, pmax = 0.0, 10
    glo = GaussianLikelihood(ndim=ndim, pmin=pmin, pmax=pmax)

    # Setup sampler
    # Set the start position and the covariance
    p0 = np.random.uniform(pmin, pmax, ndim)
    cov = np.eye(ndim) * 0.1**2

    sampler = PTMCMCSampler.PTSampler(ndim, glo.lnlikefn, glo.lnpriorfn, np.copy(cov), outDir=output_dir)

    # Add custom jump
    ujump = UniformJump(pmin, pmax)
    sampler.addProposalToCycle(ujump.jump, 5)

    # Run Sampler for 100,000 steps
    sampler.sample(
        p0,
        100000,
        burn=500,
        thin=1,
        covUpdate=500,
        SCAMweight=20,
        AMweight=20,
        DEweight=20,
    )


if __name__ == "__main__":
    """
    Command line program to run a single parallel tempering chain.
    """

    # Get info about the MPI runtime environment
    comm = MPI.COMM_WORLD
    n_chains = comm.Get_size()  # total number of chains / MPI processes
    process_rank = comm.Get_rank()  # ID of *this* chain / MPI process (zero-indexed)

    # Configure program arguments.
    parser = argparse.ArgumentParser(
        description="A wrapper script to run PTMCMCSampler's parallel tempering example using " "mpirun"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        default=None,
        help="Path of the output directory for run artifacts",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        type=int,
        default=None,
        # If any random seeds are provided, require a seed for each chain.
        nargs=n_chains,
        help="A list of integer random seeds to pass to each temperature chain, "
        "respectively.  Values must be convertible to 32-bit unsigned "
        "integers for compatability with NumPy's random.seed()",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)

    # Parse the input.
    args = parser.parse_args()

    # Seed random state for this process/chain.
    seeds = args.seeds
    if seeds:
        # Since PTMCMCSampler uses np.random*() functions to get random numbers, and we're
        # running each chain in a separate Python process, we need to seed each process to get
        # predictable results.
        # Note we only set random state here since any Python client importing this module directly
        # should retain control of random state, and only seed it once per process.

        # extract & set the seed for this process
        seed = seeds[process_rank]
        np.random.seed(seed)
        random.seed(seed)

    # Configure logging for this process.
    debug = args.debug
    if debug:
        # Get info about the MPI runtime environment
        comm = MPI.COMM_WORLD
        n_chains = comm.Get_size()  # total number of chains / MPI processes
        process_rank = comm.Get_rank()  # ID of *this* chain / MPI process (zero-indexed)

        # Configure logging in this process to DEBUG level
        # & prepend the chain # to all messages
        chain = f"Chain {process_rank + 1} of {n_chains}:"
        logging.basicConfig(
            format=f"%(levelname)s:%(asctime)s:{chain}:%(name)s: %(message)s",
            level=logging.DEBUG,
            datefmt="%H:%M:%S",
        )

    # Do the work!
    logger.debug("Starting")
    parallel_tempering_opt(args.output_dir, args.verbose)
