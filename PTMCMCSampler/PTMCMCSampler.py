import os
import sys
import time

import numpy as np

from .nutsjump import HMCJump, MALAJump, NUTSJump

try:
    from mpi4py import MPI
except ImportError:
    print("Optional mpi4py package is not installed.  MPI support is not available.")
    from . import nompi4py as MPI

try:
    import acor
except ImportError:
    # Don't complain if not available.  If you set neff, you'll get an error.  Otherwise
    # it doesn't matter.
    #    print(
    #        "Optional acor package is not installed. Acor is optionally used to calculate the "
    #        "effective chain length for output in the chain file."
    #    )
    pass


def shift_array(arr, num, fill_value=0.0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class PTSampler(object):
    """
    Parallel Tempering Markov Chain Monte-Carlo (PTMCMC) sampler.
    This implementation uses an adaptive jump proposal scheme
    by default using both standard and single component Adaptive
    Metropolis (AM) and Differential Evolution (DE) jumps.

    This implementation also makes use of MPI (mpi4py) to run
    the parallel chains.

    Along with the AM and DE jumps, the user can add custom
    jump proposals with the ``addProposalToCycle`` function.

    The sampler also supports model-switching when tuples of
    log likelihood and log prior functions are supplied. Model switch mode
    requires an MPI communicator of size 1 and does not support gradient
    arguments.

    @param ndim: number of dimensions in problem
    @param logl: single log-likelihood function or tuple of log-likelihood
    functions if using model-switching
    @param logp: single log prior function (must be normalized for evidence
    evaluation) or tuple of log prior functions if using model-switching
    @param cov: Initial covariance matrix of model parameters for jump proposals
    @param groups: Optional list of parameter groups for which to perform
    adaptive jumps. If not supplied, all parameters are grouped together.
    @param loglargs: any additional arguments (apart from the parameter vector)
    for log likelihood
    @param loglkwargs: any additional keyword arguments (apart from the parameter
    vector) for log likelihood
    @param logpargs: any additional arguments (apart from the parameter vector)
    for log prior
    @param logpkwargs: any additional keyword arguments (apart from the parameter
    vector) for log prior
    @param logl_grad: log-likelihood function, including gradients
    @param logp_grad: prior function, including gradients
    @param comm: MPI communicator used to coordinate chains
    @param outDir: Full path to output directory for chain files (default = ./chains)
    @param verbose: Update current run-status to the screen (default=True)
    @param resume: Resume from a previous chain (still in testing so beware)
    (default=False)
    @param seed: Random seed used to initialize the sampler

    """

    def __init__(
        self,
        ndim,
        logl,
        logp,
        cov,
        groups=None,
        loglargs=[],
        loglkwargs={},
        logpargs=[],
        logpkwargs={},
        logl_grad=None,
        logp_grad=None,
        comm=MPI.COMM_WORLD,
        outDir="./chains",
        verbose=True,
        resume=False,
        seed=None,
    ):
        # MPI initialization
        self.comm = comm
        self.MPIrank = self.comm.Get_rank()
        self.nchain = self.comm.Get_size()

        if self.MPIrank == 0:
            ss = np.random.SeedSequence(seed)
            child_seeds = ss.generate_state(self.nchain)
            self.stream = [np.random.default_rng(s) for s in child_seeds]
        else:
            self.stream = None
        self.stream = self.comm.scatter(self.stream, root=0)

        self.ndim = ndim

        # Model-switch mode is invoked when logl and logp are both tuples
        logl_is_tuple = isinstance(logl, tuple)
        logp_is_tuple = isinstance(logp, tuple)

        # Must match: either both tuples (modelswitch) or both single callables (normal sampling)
        if logl_is_tuple != logp_is_tuple:
            raise ValueError(
                "Model-switching requires BOTH logl and logp to be tuples. "
                "You provided a tuple for one but not the other."
            )

        self.modelswitch = logl_is_tuple

        if self.modelswitch:
            if self.nchain > 1:
                raise ValueError(
                    "Model-switch mode currently supports only a single chain; "
                    f"MPI communicator size is {self.nchain}."
                )

            # This code assumes exactly two models
            if len(logl) != 2 or len(logp) != 2:
                raise ValueError("For model-switching, logl and logp must be tuples of length 2.")

            # Tuple index 0 is treated as model 1, and tuple index 1 is treated as model 2
            self.logl1 = _function_wrapper(logl[0], loglargs, loglkwargs)
            self.logl2 = _function_wrapper(logl[1], loglargs, loglkwargs)
            self.logp1 = _function_wrapper(logp[0], logpargs, logpkwargs)
            self.logp2 = _function_wrapper(logp[1], logpargs, logpkwargs)

        else:
            self.logl = _function_wrapper(logl, loglargs, loglkwargs)
            self.logp = _function_wrapper(logp, logpargs, logpkwargs)

        if self.modelswitch and (logl_grad is not None or logp_grad is not None):
            raise ValueError("Gradients are not yet implemented in model-switch mode.")

        if logl_grad is not None and logp_grad is not None:
            self.logl_grad = _function_wrapper(logl_grad, loglargs, loglkwargs)
            self.logp_grad = _function_wrapper(logp_grad, logpargs, logpkwargs)
        else:
            self.logl_grad = None
            self.logp_grad = None

        self.outDir = outDir
        self.verbose = verbose
        self.resume = resume

        # setup output file
        if not os.path.exists(self.outDir):
            try:
                os.makedirs(self.outDir)
            except OSError:
                pass

        # find indices for which to perform adaptive jumps
        self.groups = groups
        if groups is None:
            self.groups = [np.arange(0, self.ndim)]

        # set up covariance matrix
        self.cov = cov
        self.U = [[]] * len(self.groups)
        self.S = [[]] * len(self.groups)

        # do svd on parameter groups
        for ct, group in enumerate(self.groups):
            covgroup = np.zeros((len(group), len(group)))
            for ii in range(len(group)):
                for jj in range(len(group)):
                    covgroup[ii, jj] = self.cov[group[ii], group[jj]]

            self.U[ct], self.S[ct], v = np.linalg.svd(covgroup)

        self.M2 = np.zeros((ndim, ndim))
        self.mu = np.zeros(ndim)

        # initialize proposal cycle
        self.propCycle = []
        self.jumpDict = {}

        # indicator for auxilary jumps
        self.aux = []

    def initialize(
        self,
        Niter,
        ladder=None,
        Tmin=1,
        Tmax=None,
        Tskip=100,
        isave=1000,
        covUpdate=1000,
        SCAMweight=30,
        AMweight=20,
        DEweight=50,
        NUTSweight=20,
        HMCweight=20,
        MALAweight=0,
        burn=50000,
        HMCstepsize=0.1,
        HMCsteps=300,
        maxIter=None,
        thin=10,
        i0=0,
        neff=None,
        writeHotChains=False,
        hotChain=False,
        betaSchedule=None,
        holdIter=0,
        nameChainTemps=None,
    ):
        """
        Initialize MCMC quantities

        @param Niter: Number of iterations to use for T = 1 chain. If
        betaSchedule is supplied, this is replaced by len(betaSchedule)-1
        after any holdIter plateau is prepended.
        @param ladder: User defined temperature/beta ladder. If entries are
        greater than 1 they are interpreted as temperatures and converted to beta.
        @param Tmin: Minimum temperature in ladder (default=1)
        @param Tmax: Maximum temperature in ladder (default=None)
        @param Tskip: Number of steps between proposed temperature swaps
        (default=100)
        @param isave: Write to file every isave iterations (default=1000).
        isave and thin must be positive integers, with isave a multiple of thin.
        Resuming preserves isave and requires 1 + k * (isave // thin) saved
        rows for a nonnegative integer k
        @param covUpdate: Number of iterations between AM covariance updates
        (default=1000)
        @param SCAMweight: Weight of SCAM jumps in overall jump cycle
        (default=30)
        @param AMweight: Weight of AM jumps in overall jump cycle (default=20)
        @param DEweight: Weight of DE jumps in overall jump cycle (default=50)
        @param NUTSweight: Weight of the NUTS jumps in jump cycle (default=20)
        @param MALAweight: Weight of the MALA jumps in jump cycle (default=0)
        @param HMCweight: Weight of the HMC jumps in jump cycle (default=20)
        @param HMCstepsize: Step-size of the HMC jumps (default=0.1)
        @param HMCsteps: Maximum number of steps in an HMC trajectory
        (default=300)
        @param burn: Burn in time (DE jumps added after this iteration)
        (default=50000)
        @param maxIter: Maximum number of iterations for high temperature chains
        (default=Niter)
        @param thin: MCMC samples are recorded every thin samples
        @param i0: Iteration to start MCMC (if i0 != 0, do not re-initialize)
        @param neff: Number of effective samples to collect before terminating
        @param writeHotChains: If True, write hot chains to disk
        @param hotChain: If True, include a beta=0 hot chain
        @param betaSchedule: Optional sequence of inverse temperatures/betas,
        interpreted as one beta value per sampler state. Only supported for
        model-switching runs. If supplied, the run uses a single chain with beta
        changing deterministically by schedule state, parallel tempering is
        disabled, and ladder/hotChain are not used. Values must lie in [0, 1].
        @param holdIter: Number of initial beta=0 schedule states to prepend
        before following betaSchedule.
        @param nameChainTemps: If True, name chains by temperature; if False,
        name chains by beta. If None (default), use temperature names for
        ordinary PT and beta names for model switch. Scheduled beta runs
        always use chain_schedule.txt

        """
        # Scheduled-beta mode uses an explicit beta value for each sampler state
        self.betaSchedule = betaSchedule

        if holdIter < 0:
            raise ValueError("holdIter must be >= 0")

        # A beta schedule is a single-chain model-switch mode, so reject
        # incompatible options before building the schedule
        if self.betaSchedule is not None:
            if not self.modelswitch:
                raise ValueError("betaSchedule is only supported for model-switching runs")
            if hotChain:
                raise ValueError("hotChain is not compatible with betaSchedule runs")
            if ladder is not None:
                raise ValueError("betaSchedule is not compatible with ladder being set")

            # Prepend a beta=0 flat section before the user schedule when an initial hold is requested
            self.betaSchedule = np.concatenate(
                [np.zeros(int(holdIter), dtype=float), np.asarray(self.betaSchedule, dtype=float)]
            )

            if self.betaSchedule.size < 2:
                raise ValueError("betaSchedule must contain at least two states")

            if (
                not np.all(np.isfinite(self.betaSchedule))
                or np.min(self.betaSchedule) < 0.0
                or np.max(self.betaSchedule) > 1.0
            ):
                raise ValueError("betaSchedule values must be finite and lie in [0, 1]")

            self.beta = float(self.betaSchedule[0])

            # The schedule gives beta values for states, so the number of transitions is len(schedule) - 1
            Niter = int(self.betaSchedule.size) - 1
            maxIter = Niter

        # Default maxIter for non-scheduled runs
        if maxIter is None:
            maxIter = Niter

        for name, value in (("isave", isave), ("thin", thin)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        isave = int(isave)
        thin = int(thin)

        if isave % thin != 0:
            raise ValueError(f"isave = {isave} is not a multiple of thin = {thin}")

        if Niter % thin != 0:
            print(
                "Niter = %d is not a multiple of thin = %d.  The last %d samples will be lost"
                % (Niter, thin, Niter % thin)
            )

        self.ladder = ladder
        self.covUpdate = covUpdate
        self.SCAMweight = SCAMweight
        self.AMweight = AMweight
        self.DEweight = DEweight
        self.burn = burn
        self.Tskip = Tskip
        self.thin = thin
        self.isave = isave
        self.Niter = Niter
        self.neff = neff
        self.tstart = 0

        N = int(maxIter / thin) + 1  # first sample + those we generate

        self._lnprob = np.zeros(N)
        self._lnlike = np.zeros(N)
        self._chain = np.zeros((N, self.ndim))
        self._beta = np.zeros(N)
        self.ind_next_write = 0  # Next index in these arrays to write out
        self.naccepted = 0
        self.swapProposed = 0
        self.nswap_accepted = 0

        # Number of metaparameters written after the sampled parameters.
        # Model switch output includes beta as its first metaparameter.
        self.n_metaparams = 9 if self.modelswitch else 4

        if self.modelswitch:
            self._lnprob1 = np.zeros(N)
            self._lnlike1 = np.zeros(N)
            self._lnprob2 = np.zeros(N)
            self._lnlike2 = np.zeros(N)

        # set up covariance matrix and DE buffers
        if self.MPIrank == 0:
            self._AMbuffer = np.zeros((self.covUpdate, self.ndim))
            self._DEbuffer = np.zeros((self.burn, self.ndim))

        # ##### setup default jump proposal distributions ##### #

        # Gradient-based jumps
        if self.logl_grad is not None and self.logp_grad is not None:
            # DOES MALA do anything with the burnin? (Not adaptive enabled yet)
            malajump = MALAJump(self.logl_grad, self.logp_grad, self.cov, self.burn)
            self.addProposalToCycle(malajump, MALAweight)
            if MALAweight > 0:
                print("WARNING: MALA jumps are not working properly yet")

            # Perhaps have an option to adaptively tune the mass matrix?
            # Now that is done by defaulk
            hmcjump = HMCJump(
                self.logl_grad,
                self.logp_grad,
                self.cov,
                self.burn,
                stepsize=HMCstepsize,
                nminsteps=2,
                nmaxsteps=HMCsteps,
            )
            self.addProposalToCycle(hmcjump, HMCweight)

            # Target acceptance rate (delta) should be optimal for 0.6
            nutsjump = NUTSJump(
                self.logl_grad,
                self.logp_grad,
                self.cov,
                self.burn,
                trajectoryDir=None,
                write_burnin=False,
                force_trajlen=None,
                force_epsilon=None,
                delta=0.6,
            )
            self.addProposalToCycle(nutsjump, NUTSweight)

        # add SCAM
        self.addProposalToCycle(self.covarianceJumpProposalSCAM, self.SCAMweight)

        # add AM
        self.addProposalToCycle(self.covarianceJumpProposalAM, self.AMweight)

        # check length of jump cycle
        if len(self.propCycle) == 0:
            raise ValueError("No jump proposals specified!")

        # randomize cycle
        self.randomizeProposalCycle()

        # Ladder setup
        if self.betaSchedule is not None:
            # varying-beta run: no PT ladder
            self.ladder = np.array([1.0])
        else:
            # If a ladder is supplied, accept either temperatures (>1) or betas (<=1).
            if self.ladder is not None and len(self.ladder) > 0:
                self.ladder = np.asarray(self.ladder, dtype=float)
                if self.ladder.ndim != 1 or self.ladder.size == 0:
                    raise ValueError("ladder must be a one-dimensional sequence")
                if not np.all(np.isfinite(self.ladder)):
                    raise ValueError("ladder contains non-finite values")
                if np.max(self.ladder) > 1.0:
                    self.ladder = 1.0 / self.ladder
                if np.min(self.ladder) < 0.0 or np.max(self.ladder) > 1.0:
                    raise ValueError("beta ladder values must lie in [0, 1]")

            # ladder not specified, create beta ladder from the original temperature inputs
            else:
                self.ladder = self.Ladder(Tmin, Tmax=Tmax)

            # beta for current chain (only meaningful for PT runs)
            if hotChain and self.nchain > 1:
                self.ladder[-1] = 0.0
            self.beta = self.ladder[self.MPIrank]

        # Default to temperature names for ordinary PT and beta names for model-switch
        if nameChainTemps is None:
            nameChainTemps = not self.modelswitch

        # Name chain files
        if self.betaSchedule is not None:
            # beta changes over time, fixed filename for scheduled runs
            self.fname = self.outDir + "/chain_schedule.txt"
        else:
            if hotChain and self.MPIrank == self.nchain - 1:
                self.beta = 0  # This is the "hot chain"
                if nameChainTemps:  # name chains by temperature
                    self.fname = self.outDir + "/chain_hot.txt"
                else:  # name chains by beta
                    self.fname = self.outDir + "/chain_0.txt"

            elif nameChainTemps:  # name chains by temperature
                self.fname = self.outDir + "/chain_{0}.txt".format(1 / self.beta)

            else:  # name chains by beta
                self.fname = self.outDir + "/chain_{0}.txt".format(self.beta)

        # write hot chains
        self.writeHotChains = writeHotChains

        self.resumeLength = 0
        if self.resume and os.path.isfile(self.fname):
            if self.verbose:
                print("Resuming run from chain file {0}".format(self.fname))
            try:
                self.resumechain = np.loadtxt(self.fname, ndmin=2)
                expected_cols = self.ndim + self.n_metaparams
                if self.resumechain.shape[1] != expected_cols:
                    current_mode = "model-switch" if self.modelswitch else "non-model-switch"
                    expected_format = (
                        "parameters + 9 metaparameters" if self.modelswitch else "parameters + 4 metaparameters"
                    )
                    raise Exception(
                        f"Cannot resume chain file {self.fname}: expected {expected_cols} columns "
                        f"({expected_format}) for {current_mode}, but found {self.resumechain.shape[1]}. "
                        "This usually means the chain file was created with a different resume/output format."
                    )
                self.resumeLength = self.resumechain.shape[0]  # Number of samples read from old chain
            except ValueError as error:
                print("Reading old chain files failed with error", error)
                raise Exception("Couldn't read old chain to resume")

            if self.betaSchedule is not None:
                saved_betas = self.resumechain[:, self.ndim]
                expected_betas = self.betaSchedule[:: self.thin]

                if self.resumeLength > len(expected_betas) or not np.allclose(
                    saved_betas, expected_betas[: self.resumeLength]
                ):
                    raise ValueError("Saved beta values do not match betaSchedule")
            block_rows = self.isave // self.thin
            if self.resumeLength < 1 or (self.resumeLength - 1) % block_rows != 0:
                raise ValueError(
                    f"Old chain has {self.resumeLength} rows; expected "
                    f"1 + k * (isave // thin) = 1 + k * {block_rows} "
                    "rows for a nonnegative integer k."
                )
            self._chainfile = open(self.fname, "a")
            print(
                "Resuming with",
                self.resumeLength,
                "samples from file representing",
                (self.resumeLength - 1) * self.thin + 1,
                "original samples",
            )
        else:
            self._chainfile = open(self.fname, "w")
        self._chainfile.close()

    def updateChains(
        self,
        p0,
        lnlike0,
        lnprob0,
        iter,
        lnlike1=None,
        lnprob1=None,
        lnlike2=None,
        lnprob2=None,
    ):
        """
        Update chains after jump proposals.

        @param p0: Current parameter vector
        @param lnlike0: Current log-likelihood value
        @param lnprob0: Current log posterior value
        @param iter: Current iteration number
        @param lnlike1: Current model 1 log-likelihood value, if using
        model-switching
        @param lnprob1: Current model 1 log posterior value, if using
        model-switching
        @param lnlike2: Current model 2 log-likelihood value, if using
        model-switching
        @param lnprob2: Current model 2 log posterior value, if using
        model-switching

        """
        # update buffer
        if self.MPIrank == 0:
            self._AMbuffer[iter % self.covUpdate, :] = p0

        # put results into arrays
        if iter % self.thin == 0:
            ind = int(iter / self.thin)
            self._chain[ind, :] = p0
            # In model switch output, beta is written as the first metaparameter
            self._beta[ind] = self.beta
            self._lnlike[ind] = lnlike0
            self._lnprob[ind] = lnprob0

            # Use None checks rather than truthiness so zero valued log probabilities are still stored
            if (lnlike1 is not None) and (lnlike2 is not None) and (lnprob1 is not None) and (lnprob2 is not None):
                self._lnlike1[ind] = lnlike1
                self._lnprob1[ind] = lnprob1
                self._lnlike2[ind] = lnlike2
                self._lnprob2[ind] = lnprob2

        # write to file
        if iter % self.isave == 0:
            self.writeOutput(iter)

    def writeOutput(self, iter):
        """
        Write chains and covariance matrix. Called every isave on samples or at end.

        @param iter: Iteration of sampler

        """
        if iter // self.thin >= self.ind_next_write:

            if self.writeHotChains or self.MPIrank == 0:
                self._writeToFile(iter)

            # write output covariance matrix
            if iter > 0:
                np.save(self.outDir + "/cov.npy", self.cov)

            if self.MPIrank == 0 and self.verbose:
                if iter > 0:
                    sys.stdout.write("\r")
                percent = iter / self.Niter * 100  # Percent of total work finished
                acceptance = self.naccepted / iter if iter > 0 else 0
                elapsed = time.time() - self.tstart
                if self.resume:
                    # Percentage of new work done
                    if self.Niter <= self.resumeLength * self.thin:
                        percentnew = 100
                    else:
                        percentnew = (
                            (iter - self.resumeLength * self.thin) / (self.Niter - self.resumeLength * self.thin) * 100
                        )
                    sys.stdout.write(
                        "Finished %2.2f percent (%2.2f percent of new work) in %f s Acceptance rate = %g"
                        % (percent, percentnew, elapsed, acceptance)
                    )
                else:
                    sys.stdout.write(
                        "Finished %2.2f percent in %f s Acceptance rate = %g" % (percent, elapsed, acceptance)
                    )
                sys.stdout.flush()

    def sample(
        self,
        p0,
        Niter,
        ladder=None,
        Tmin=1,
        Tmax=None,
        Tskip=100,
        isave=1000,
        covUpdate=1000,
        SCAMweight=20,
        AMweight=20,
        DEweight=20,
        NUTSweight=20,
        MALAweight=20,
        HMCweight=20,
        burn=10000,
        HMCstepsize=0.1,
        HMCsteps=300,
        maxIter=None,
        thin=10,
        i0=0,
        neff=None,
        writeHotChains=False,
        hotChain=False,
        betaSchedule=None,
        holdIter=0,
        nameChainTemps=None,
    ):
        """
        Function to carry out PTMCMC sampling.

        @param p0: Initial parameter vector
        @param Niter: Number of iterations to use for T = 1 chain. If
        betaSchedule is supplied, the schedule length determines the run length.
        @param ladder: User defined temperature/beta ladder. If entries are
        greater than 1 they are interpreted as temperatures and converted to beta.
        @param Tmin: Minimum temperature in ladder (default=1)
        @param Tmax: Maximum temperature in ladder (default=None)
        @param Tskip: Number of steps between proposed temperature swaps
        (default=100)
        @param isave: Write to file every isave iterations (default=1000).
        isave and thin must be positive integers, with isave a multiple of thin.
        Resuming preserves isave and requires 1 + k * (isave // thin) saved
        rows for a nonnegative integer k
        @param covUpdate: Number of iterations between AM covariance updates
        (default=1000)
        @param SCAMweight: Weight of SCAM jumps in overall jump cycle
        (default=20)
        @param AMweight: Weight of AM jumps in overall jump cycle (default=20)
        @param DEweight: Weight of DE jumps in overall jump cycle (default=20)
        @param NUTSweight: Weight of the NUTS jumps in jump cycle (default=20)
        @param MALAweight: Weight of the MALA jumps in jump cycle (default=20)
        @param HMCweight: Weight of the HMC jumps in jump cycle (default=20)
        @param HMCstepsize: Step-size of the HMC jumps (default=0.1)
        @param HMCsteps: Maximum number of steps in an HMC trajectory
        (default=300)
        @param burn: Burn in time (DE jumps added after this iteration)
        (default=10000)
        @param maxIter: Maximum number of iterations for high temperature chains
        (default=Niter)
        @param thin: MCMC samples are recorded every thin samples
        @param i0: Iteration to start MCMC (if i0 != 0, do not re-initialize)
        @param neff: Number of effective samples to collect before terminating
        @param writeHotChains: If True, write hot chains to disk
        @param hotChain: If True, include a beta=0 hot chain
        @param betaSchedule: Optional sequence of inverse temperatures/betas,
        interpreted as one beta value per sampler state. Only supported for
        model-switching runs. If supplied, the sampler performs a varying-beta
        run rather than standard parallel tempering. Parallel tempering swaps are
        disabled, only single-chain runs are supported, ladder and hotChain
        cannot be used, and the number of transition steps is
        len(self.betaSchedule)-1
        @param holdIter: Number of initial beta=0 schedule states to prepend
        before following betaSchedule.
        @param nameChainTemps: If True, name chains by temperature. If False,
        name chains by beta. If None (default), use temperature names for
        ordinary PT and beta names for model switch. Scheduled beta runs
        always use chain_schedule.txt

        """

        # set up arrays to store lnprob, lnlike and chain
        # if picking up from previous run, don't re-initialize
        if i0 == 0:
            self.initialize(
                Niter,
                ladder=ladder,
                Tmin=Tmin,
                Tmax=Tmax,
                Tskip=Tskip,
                isave=isave,
                covUpdate=covUpdate,
                SCAMweight=SCAMweight,
                AMweight=AMweight,
                DEweight=DEweight,
                NUTSweight=NUTSweight,
                MALAweight=MALAweight,
                HMCweight=HMCweight,
                burn=burn,
                HMCstepsize=HMCstepsize,
                HMCsteps=HMCsteps,
                maxIter=maxIter,
                thin=thin,
                i0=i0,
                neff=neff,
                writeHotChains=writeHotChains,
                betaSchedule=betaSchedule,
                holdIter=holdIter,
                hotChain=hotChain,
                nameChainTemps=nameChainTemps,
            )

        # compute lnprob for initial point in chain

        # if resuming, just start with first point in chain
        if self.resume and self.resumeLength > 0:
            if self.modelswitch:
                p0 = self.resumechain[0, : self.ndim]
                self.beta = self.resumechain[0, self.ndim]
                lnprob0 = self.resumechain[0, self.ndim + 1]
                lnlike0 = self.resumechain[0, self.ndim + 2]
                lnprob1 = self.resumechain[0, self.ndim + 3]
                lnlike1 = self.resumechain[0, self.ndim + 4]
                lnprob2 = self.resumechain[0, self.ndim + 5]
                lnlike2 = self.resumechain[0, self.ndim + 6]

            else:
                p0, lnlike0, lnprob0 = (
                    self.resumechain[0, :-4],
                    self.resumechain[0, -3],
                    self.resumechain[0, -4],
                )

            self.ind_next_write = self.resumeLength

        else:
            # compute prior and likelihood
            if not self.modelswitch:
                lp = self.logp(p0)

                if lp == -np.inf:
                    lnlike0 = -np.inf
                    lnprob0 = -np.inf

                else:
                    lnlike0 = self.logl(p0)
                    if lnlike0 == -np.inf:
                        lnprob0 = -np.inf
                    else:
                        lnprob0 = self.beta * lnlike0 + lp

            else:

                lp1 = self.logp1(p0)
                lp2 = self.logp2(p0)

                if lp1 == -np.inf or lp2 == -np.inf:
                    lnprob0 = -np.inf
                    lnlike0 = -np.inf
                    lnlike1 = -np.inf
                    lnprob1 = -np.inf
                    lnlike2 = -np.inf
                    lnprob2 = -np.inf

                else:
                    lnlike1 = self.logl1(p0)
                    lnlike2 = self.logl2(p0)

                    if lnlike1 == -np.inf or lnlike2 == -np.inf:
                        lnprob0 = -np.inf
                        lnlike0 = -np.inf
                        lnlike1 = -np.inf
                        lnprob1 = -np.inf
                        lnlike2 = -np.inf
                        lnprob2 = -np.inf
                    else:
                        lnprob1 = lnlike1 + lp1
                        lnprob2 = lnlike2 + lp2

                        lnlike0 = lnprob1 - lnprob2  # Difference between the two model log posteriors
                        lnprob0 = self.beta * lnlike0 + lnprob2

        # Scheduled beta runs change the model switch target distribution each iteration
        if self.betaSchedule is not None:
            current_idx = i0
            if current_idx < 0 or current_idx >= len(self.betaSchedule):
                raise IndexError(
                    "betaSchedule index out of range at initialization: "
                    f"idx={current_idx}, len={len(self.betaSchedule)}"
                )

            self.beta = float(self.betaSchedule[current_idx])
            if (not np.isfinite(lnprob2)) or (not np.isfinite(lnlike0)):
                lnprob0 = -np.inf
            else:
                lnprob0 = self.beta * lnlike0 + lnprob2

        # record first values
        self.tstart = time.time()

        if not self.modelswitch:
            self.updateChains(p0, lnlike0, lnprob0, i0)

        else:
            self.updateChains(
                p0,
                lnlike0,
                lnprob0,
                i0,
                lnlike1=lnlike1,
                lnprob1=lnprob1,
                lnlike2=lnlike2,
                lnprob2=lnprob2,
            )

        self.comm.barrier()

        # start iterations
        iter = i0

        runComplete = False
        while runComplete is False:
            iter += 1
            self.comm.barrier()  # make sure all processes are at the same iteration
            # call PTMCMCOneStep
            if not self.modelswitch:
                p0, lnlike0, lnprob0 = self.PTMCMCOneStep(p0, lnlike0, lnprob0, iter)
            else:
                p0, lnlike0, lnprob0, lnlike1, lnprob1, lnlike2, lnprob2 = self.PTMCMCOneStep(
                    p0,
                    lnlike0,
                    lnprob0,
                    iter,
                    lnlike1=lnlike1,
                    lnprob1=lnprob1,
                    lnlike2=lnlike2,
                    lnprob2=lnprob2,
                )

            # rank 0 decides whether to stop
            if self.MPIrank == 0:
                if iter >= self.Niter:  # stop if reached maximum number of iterations
                    message = "\nRun Complete"
                    runComplete = True
                elif self.neff:  # Stop if effective number of samples reached if requested
                    if iter % 1000 == 0 and iter > 2 * self.burn and self.MPIrank == 0:
                        Neff = iter / max(
                            1,
                            np.nanmax(
                                [acor.acor(self._chain[self.burn : (iter - 1), ii])[0] for ii in range(self.ndim)]
                            ),
                        )
                        # print('\n {0} effective samples'.format(Neff))
                        if int(Neff) >= self.neff:
                            message = "\nRun Complete with {0} effective samples".format(int(Neff))
                            runComplete = True

            runComplete = self.comm.bcast(runComplete, root=0)  # rank 0 tells others whether to stop

            if runComplete:
                self.writeOutput(iter)  # Possibly write partial block
                if self.MPIrank == 0 and self.verbose:
                    print(message)

    def PTMCMCOneStep(
        self,
        p0,
        lnlike0,
        lnprob0,
        iter,
        lnlike1=None,
        lnprob1=None,
        lnlike2=None,
        lnprob2=None,
    ):
        """
        Function to carry out one PTMCMC sampling step.

        @param p0: Initial parameter vector
        @param lnlike0: Initial log-likelihood value
        @param lnprob0: Initial log probability value
        @param iter: Iteration number
        @param lnlike1: Model 1 log-likelihood value, if using model-switching
        @param lnprob1: Model 1 log posterior value, if using model-switching
        @param lnlike2: Model 2 log-likelihood value, if using model-switching
        @param lnprob2: Model 2 log posterior value, if using model-switching

        @return p0: next value of parameter vector after one MCMC step
        @return lnlike0: next value of likelihood after one MCMC step
        @return lnprob0: next value of posterior after one MCMC step

        """
        # update covariance matrix
        if (iter - 1) % self.covUpdate == 0 and (iter - 1) != 0 and self.MPIrank == 0:
            self._updateRecursive(iter - 1, self.covUpdate)

            # broadcast to other chains
            [self.comm.send(self.cov, dest=rank + 1, tag=111) for rank in range(self.nchain - 1)]

        # update covariance matrix
        if (iter - 1) % self.covUpdate == 0 and (iter - 1) != 0 and self.MPIrank > 0:
            self.cov[:, :] = self.comm.recv(source=0, tag=111)
            for ct, group in enumerate(self.groups):
                covgroup = np.zeros((len(group), len(group)))
                for ii in range(len(group)):
                    for jj in range(len(group)):
                        covgroup[ii, jj] = self.cov[group[ii], group[jj]]

                self.U[ct], self.S[ct], v = np.linalg.svd(covgroup)

        # update DE buffer
        if (iter - 1) % self.burn == 0 and (iter - 1) != 0 and self.MPIrank == 0:
            self._updateDEbuffer(iter - 1, self.burn)

            # broadcast to other chains
            [self.comm.send(self._DEbuffer, dest=rank + 1, tag=222) for rank in range(self.nchain - 1)]

        # update DE buffer
        if (iter - 1) % self.burn == 0 and (iter - 1) != 0 and self.MPIrank > 0:
            self._DEbuffer = self.comm.recv(source=0, tag=222)

            # randomize cycle
            if self.DEJump not in self.propCycle:
                self.addProposalToCycle(self.DEJump, self.DEweight)
                self.randomizeProposalCycle()

        # after burn in, add DE jumps
        if (iter - 1) == self.burn and self.MPIrank == 0:
            if self.verbose:
                print("Adding DE jump with weight {0}".format(self.DEweight))
            self.addProposalToCycle(self.DEJump, self.DEweight)

            # randomize cycle
            self.randomizeProposalCycle()

        # Scheduled beta runs change the model-switch target by state
        if self.betaSchedule is not None:
            idx = iter
            if idx < 0 or idx >= len(self.betaSchedule):
                raise IndexError(f"betaSchedule index out of range: idx={idx}, len={len(self.betaSchedule)}")
            self.beta = float(self.betaSchedule[idx])

            if (not np.isfinite(lnprob2)) or (not np.isfinite(lnlike0)):
                lnprob0 = -np.inf
            else:
                lnprob0 = self.beta * lnlike0 + lnprob2

        # jump proposal ###

        # if resuming, just use previous chain points. Use each one thin times to compensate for
        # thinning when they were written out
        if self.resume and self.resumeLength > 0 and iter < self.resumeLength * self.thin:
            row = self.resumechain[iter // self.thin]

            if self.modelswitch:
                p0 = row[: self.ndim]

                if self.betaSchedule is None:
                    self.beta = row[self.ndim]

                lnprob0 = row[self.ndim + 1]
                lnlike0 = row[self.ndim + 2]
                lnprob1 = row[self.ndim + 3]
                lnlike1 = row[self.ndim + 4]
                lnprob2 = row[self.ndim + 5]
                lnlike2 = row[self.ndim + 6]

                if self.betaSchedule is not None:
                    if (not np.isfinite(lnprob2)) or (not np.isfinite(lnlike0)):
                        lnprob0 = -np.inf
                    else:
                        lnprob0 = self.beta * lnlike0 + lnprob2

            else:
                p0, lnlike0, lnprob0 = row[:-4], row[-3], row[-4]

            # update acceptance counter
            self.naccepted = iter * row[-2]

        else:
            y, qxy, jump_name = self._jump(p0, iter)
            self.jumpDict[jump_name][0] += 1

            # compute prior and likelihood
            if not self.modelswitch:
                lp = self.logp(y)

                if lp == -np.inf:
                    newlnlike = -np.inf
                    newlnprob = -np.inf

                else:
                    newlnlike = self.logl(y)
                    if newlnlike == -np.inf:
                        newlnprob = -np.inf
                    else:
                        newlnprob = self.beta * newlnlike + lp

            else:
                lp1 = self.logp1(y)
                lp2 = self.logp2(y)

                # Set all model specific log values on invalid proposals so
                # rejected jumps do not leave undefined variables
                if lp1 == -np.inf or lp2 == -np.inf:
                    newlnlike = -np.inf
                    newlnprob = -np.inf
                    newlnlike1 = -np.inf
                    newlnprob1 = -np.inf
                    newlnlike2 = -np.inf
                    newlnprob2 = -np.inf

                else:
                    newlnlike1 = self.logl1(y)
                    newlnlike2 = self.logl2(y)

                    if newlnlike1 == -np.inf or newlnlike2 == -np.inf:
                        newlnlike = -np.inf
                        newlnprob = -np.inf
                        newlnlike1 = -np.inf
                        newlnprob1 = -np.inf
                        newlnlike2 = -np.inf
                        newlnprob2 = -np.inf
                    else:
                        newlnprob1 = newlnlike1 + lp1  # no beta here, we want full posterior of each model
                        newlnprob2 = newlnlike2 + lp2  # no beta here, we want full posterior of each model

                        newlnlike = newlnprob1 - newlnprob2

                        # ln posterior = beta * ln likelihood + ln prior
                        # ln prior is set to ln posterior of the second model
                        # ln likelihood is the difference between ln posterior of the first and second models
                        # beta determines how much of newlnprob1 vs newlnprob2
                        newlnprob = self.beta * newlnlike + newlnprob2

            # hastings step
            diff = newlnprob - lnprob0 + qxy

            rand_log = np.log(self.stream.random())
            if diff > rand_log:
                # accept jump
                p0, lnlike0, lnprob0 = y, newlnlike, newlnprob

                if self.modelswitch:
                    lnlike1, lnlike2, lnprob1, lnprob2 = (
                        newlnlike1,
                        newlnlike2,
                        newlnprob1,
                        newlnprob2,
                    )

                # update acceptance counter
                self.naccepted += 1
                self.jumpDict[jump_name][1] += 1

        # Update chains
        if self.modelswitch:
            self.updateChains(
                p0,
                lnlike0,
                lnprob0,
                iter,
                lnlike1=lnlike1,
                lnprob1=lnprob1,
                lnlike2=lnlike2,
                lnprob2=lnprob2,
            )
            return p0, lnlike0, lnprob0, lnlike1, lnprob1, lnlike2, lnprob2

        else:
            # temperature swap
            if iter % self.Tskip == 0 and self.nchain > 1:
                p0, lnlike0, lnprob0 = self.PTswap(p0, lnlike0, lnprob0, iter)

            self.updateChains(p0, lnlike0, lnprob0, iter)

            return p0, lnlike0, lnprob0

    def PTswap(self, p0, lnlike0, lnprob0, iter):
        """
        Do parallel tempering swap using inverse temperatures/betas.

        (Repurposed from Neil Cornish/Bence Becsy's code)

        Assumes self.ladder is an array of inverse temperatures/betas, with
        ladder[0] corresponding to the cold chain and subsequent entries
        corresponding to hotter chains.

        Swap acceptance rates are computed per chain by storing
        the number of swaps proposed and accepted. Since swaps
        are proposed for every chain, swapProposed is always
        incremented and nswap_accepted is incremented only
        for chains where a swap is accepted. The swap acceptance
        is calculated elsewhere.

        @param p0: current parameter vector
        @param lnlike0: current log-likelihood
        @param lnprob0: current log posterior value
        @param iter: current iteration number

        @return p0: new parameter vector
        @return lnlike0: new log-likelihood
        @return lnprob0: new log posterior value

        """
        betas = self.ladder

        log_Ls = self.comm.gather(lnlike0, root=0)  # list of likelihoods from each chain
        p0s = self.comm.gather(p0, root=0)  # list of parameter arrays from each chain
        swap_accepted = np.zeros(self.nchain)
        new_p0s = np.zeros_like(p0s)
        new_log_Ls = np.zeros_like(log_Ls)

        if self.MPIrank == 0:
            # set up map to help keep track of swaps
            swap_map = list(range(self.nchain))

            # loop through and propose a swap at each chain (starting from hottest chain and going down in T)
            # and keep track of results in swap_map
            for swap_chain in reversed(range(self.nchain - 1)):
                log_acc_ratio = -log_Ls[swap_map[swap_chain]] * betas[swap_chain]
                log_acc_ratio += -log_Ls[swap_map[swap_chain + 1]] * betas[swap_chain + 1]
                log_acc_ratio += log_Ls[swap_map[swap_chain + 1]] * betas[swap_chain]
                log_acc_ratio += log_Ls[swap_map[swap_chain]] * betas[swap_chain + 1]

                acc_ratio = np.exp(log_acc_ratio)
                if self.stream.uniform() <= acc_ratio:
                    swap_map[swap_chain], swap_map[swap_chain + 1] = swap_map[swap_chain + 1], swap_map[swap_chain]
                    swap_accepted[swap_chain] += 1

            # loop through the chains and record the new samples and log_Ls
            for j in range(self.nchain):
                new_p0s[j] = p0s[swap_map[j]]
                new_log_Ls[j] = log_Ls[swap_map[j]]

        # broadcast the new samples and log_Ls to all chains
        p0 = self.comm.scatter(new_p0s, root=0)
        lnlike0 = self.comm.scatter(new_log_Ls, root=0)
        self.nswap_accepted += self.comm.scatter(swap_accepted, root=0)
        self.swapProposed += 1

        # calculate new posterior values
        if lnlike0 == -np.inf:
            lnprob0 = -np.inf
        else:
            lnprob0 = self.beta * lnlike0 + self.logp(p0)

        return p0, lnlike0, lnprob0

    def Ladder(self, Tmin=1, Tmax=None, tstep=None):
        """
        Method to compute a geometrically spaced inverse-temperature ladder,
        where beta = 1 / T
        """

        if Tmin is None:
            Tmin = 1
        if Tmin <= 0:
            raise ValueError("Tmin must be > 0")
        if Tmax is not None and Tmax <= 0:
            raise ValueError("Tmax must be > 0")
        if Tmax is not None and Tmax < Tmin:
            raise ValueError("Tmax must be >= Tmin")

        if self.nchain > 1:
            if tstep is None and Tmax is None:
                tstep = 1 + np.sqrt(2 / self.ndim)
            elif tstep is None and Tmax is not None:
                tstep = np.exp(np.log(Tmax / Tmin) / (self.nchain - 1))

            beta0 = 1.0 / Tmin
            ladder = np.zeros(self.nchain)
            for ii in range(self.nchain):
                ladder[ii] = beta0 * tstep ** (-ii)
        else:
            ladder = np.array([1.0])

        return ladder

    def _writeToFile(self, iter):
        """
        Function to write chain file. Non-model-switch output has parameter
        values followed by 4 metaparameters: log-posterior, log-likelihood,
        acceptance rate, and PT acceptance rate. Model-switch output has
        parameter values followed by 9 metaparameters: beta, log-posterior,
        log-likelihood, model-1 log-posterior, model-1 log-likelihood,
        model-2 log-posterior, model-2 log-likelihood, acceptance rate,
        and PT acceptance rate

        @param iter: Iteration of sampler

        """

        self._chainfile = open(self.fname, "a+")
        # index 0 is the initial element.  So after 10*thin iterations we need to write elements 1..10
        write_end = iter // self.thin + 1  # First element not to write.
        for ind in range(self.ind_next_write, write_end):
            pt_acc = 1
            if self.MPIrank < self.nchain - 1 and self.swapProposed != 0:
                pt_acc = self.nswap_accepted / self.swapProposed

            # parameters always come first
            self._chainfile.write("\t".join(["%22.22f" % (self._chain[ind, kk]) for kk in range(self.ndim)]))

            # Model-switch output writes beta as the first metaparameter
            if self.modelswitch:
                self._chainfile.write("\t%22.22f" % self._beta[ind])

            # main posterior / likelihood for the active chain state
            self._chainfile.write("\t%f\t%f" % (self._lnprob[ind], self._lnlike[ind]))

            # extra model-switch diagnostics, if present
            if self.modelswitch:
                self._chainfile.write(
                    "\t%f\t%f\t%f\t%f"
                    % (
                        self._lnprob1[ind],
                        self._lnlike1[ind],
                        self._lnprob2[ind],
                        self._lnlike2[ind],
                    )
                )

            # acceptance metadata goes last
            self._chainfile.write("\t%f\t%f\n" % (self.naccepted / iter if iter > 0 else 0, pt_acc))
        self._chainfile.close()
        self.ind_next_write = write_end  # Ready for next write

        # write jump statistics files ####

        # only for T=1 chain
        if self.MPIrank == 0:

            # first write file contaning jump names and jump rates
            fout = open(self.outDir + "/jumps.txt", "w")
            njumps = len(self.propCycle)
            ujumps = np.array(list(set(self.propCycle)))
            for jump in ujumps:
                fout.write("%s %4.2g\n" % (jump.__name__, np.sum(np.array(self.propCycle) == jump) / njumps))

            fout.close()

            # now write jump statistics for each jump proposal
            for jump in self.jumpDict:
                fout = open(self.outDir + "/" + jump + "_jump.txt", "a+")
                fout.write("%g\n" % (self.jumpDict[jump][1] / max(1, self.jumpDict[jump][0])))
                fout.close()

    # function to update covariance matrix for jump proposals
    def _updateRecursive(self, iter, mem):
        """
        Function to recursively update sample covariance matrix.

        @param iter: Iteration of sampler
        @param mem: Number of steps between updates

        """
        it = iter - mem
        ndim = self.ndim

        if it == 0:
            self.M2 = np.zeros((ndim, ndim))
            self.mu = np.zeros(ndim)

        for ii in range(mem):
            diff = np.zeros(ndim)
            it += 1
            for jj in range(ndim):

                diff[jj] = self._AMbuffer[ii, jj] - self.mu[jj]
                self.mu[jj] += diff[jj] / it

            self.M2 += np.outer(diff, (self._AMbuffer[ii, :] - self.mu))

        self.cov[:, :] = self.M2 / (it - 1)

        # do svd on parameter groups
        for ct, group in enumerate(self.groups):
            covgroup = np.zeros((len(group), len(group)))
            for ii in range(len(group)):
                for jj in range(len(group)):
                    covgroup[ii, jj] = self.cov[group[ii], group[jj]]

            self.U[ct], self.S[ct], v = np.linalg.svd(covgroup)

    # update DE buffer samples
    def _updateDEbuffer(self, iter, burn):
        """
        Update Differential Evolution with last burn values in the total chain.

        @param iter: Iteration of sampler
        @param burn: Total number of samples in DE buffer

        """

        self._DEbuffer = shift_array(self._DEbuffer, -len(self._AMbuffer))  # shift DEbuffer to the left
        self._DEbuffer[-len(self._AMbuffer) :] = self._AMbuffer  # add new samples to the new empty spaces

    # SCAM jump
    def covarianceJumpProposalSCAM(self, x, iter, beta):
        """
        Single Component Adaptive Jump Proposal. This function will occasionally
        jump in more than 1 parameter. It will also occasionally use different
        jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return q: New position in parameter space
        @return qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # choose group
        jumpind = self.stream.integers(0, len(self.groups))
        ndim = len(self.groups[jumpind])

        # adjust step size
        prob = self.stream.random()

        # large jump
        if prob > 0.97:
            scale = 10

        # small jump
        elif prob > 0.9:
            scale = 0.2

        # small-medium jump
        # elif prob > 0.6:
        #   scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        # adjust scale based on temperature
        if (not self.modelswitch) and beta >= 0.01:
            scale *= np.sqrt(1 / beta)

        # get parmeters in new diagonalized basis
        # y = np.dot(self.U.T, x[self.covinds])

        # make correlated componentwise adaptive jump
        ind = np.unique(self.stream.integers(0, ndim, 1))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        q[self.groups[jumpind]] += (
            self.stream.standard_normal() * cd * np.sqrt(self.S[jumpind][ind]) * self.U[jumpind][:, ind].flatten()
        )

        return q, qxy

    # AM jump
    def covarianceJumpProposalAM(self, x, iter, beta):
        """
        Adaptive Jump Proposal. This function will occasionally
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return q: New position in parameter space
        @return qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # choose group
        jumpind = self.stream.integers(0, len(self.groups))

        # adjust step size
        prob = self.stream.random()

        # large jump
        if prob > 0.97:
            scale = 10

        # small jump
        elif prob > 0.9:
            scale = 0.2

        # small-medium jump
        # elif prob > 0.6:
        #    scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        # adjust scale based on temperature
        if (not self.modelswitch) and beta >= 0.01:
            scale *= np.sqrt(1 / beta)

        # get parmeters in new diagonalized basis
        y = np.dot(self.U[jumpind].T, x[self.groups[jumpind]])

        # make correlated componentwise adaptive jump
        ind = np.arange(len(self.groups[jumpind]))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        y[ind] = y[ind] + self.stream.standard_normal(neff) * cd * np.sqrt(self.S[jumpind][ind])
        q[self.groups[jumpind]] = np.dot(self.U[jumpind], y)

        return q, qxy

    # Differential evolution jump
    def DEJump(self, x, iter, beta):
        """
        Differential Evolution Jump. This function will occasionally
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return q: New position in parameter space
        @return qxy: Forward-Backward jump probability

        """

        # get old parameters
        q = x.copy()
        qxy = 0

        # choose group
        jumpind = self.stream.integers(0, len(self.groups))
        ndim = len(self.groups[jumpind])

        bufsize = len(self._DEbuffer)

        # draw a random integer from 0 - iter
        mm = self.stream.integers(0, bufsize)
        nn = self.stream.integers(0, bufsize)

        # make sure mm and nn are not the same iteration
        while mm == nn:
            nn = self.stream.integers(0, bufsize)

        # get jump scale size
        prob = self.stream.random()

        # mode jump
        if prob > 0.5 or self.modelswitch or beta <= 0.0:
            scale = 1.0
        else:
            scale = self.stream.random() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1 / beta)

        for ii in range(ndim):

            # jump size
            sigma = self._DEbuffer[mm, self.groups[jumpind][ii]] - self._DEbuffer[nn, self.groups[jumpind][ii]]

            # jump
            q[self.groups[jumpind][ii]] += scale * sigma

        return q, qxy

    # add jump proposal distribution functions
    def addProposalToCycle(self, func, weight):
        """
        Add jump proposal distributions to cycle with a given weight.

        @param func: jump proposal function
        @param weight: jump proposal function weight in cycle

        """

        # get length of cycle so far
        length = len(self.propCycle)

        # check for 0 weight
        if weight == 0:
            # print('ERROR: Can not have 0 weight in proposal cycle!')
            # sys.exit()
            return

        # add proposal to cycle
        for ii in range(length, length + weight):
            self.propCycle.append(func)

        # add to jump dictionary and initialize file
        if func.__name__ not in self.jumpDict:
            self.jumpDict[func.__name__] = [0, 0]
            fout = open(self.outDir + "/" + func.__name__ + "_jump.txt", "w")
            fout.close()

    # add auxilary jump proposal distribution functions
    def addAuxilaryJump(self, func):
        """
        Add auxilary jump proposal distribution. This will be called after every
        standard jump proposal. Examples include cyclic boundary conditions and
        pulsar phase fixes.

        @param func: jump proposal function

        """

        # set auxilary jump
        self.aux.append(func)

    # randomized proposal cycle
    def randomizeProposalCycle(self):
        """
        Randomize proposal cycle that has already been filled

        """

        # get length of full cycle
        length = len(self.propCycle)

        # get random integers
        index = np.arange(length)
        self.stream.shuffle(index)

        # randomize proposal cycle
        self.randomizedPropCycle = [self.propCycle[ind] for ind in index]

    # call proposal functions from cycle
    def _jump(self, x, iter):
        """
        Call Jump proposals.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler

        @return q: New position in parameter space
        @return qxy: Forward-Backward jump probability
        @return jump_name: Name of proposal used

        """

        # get length of cycle
        length = len(self.propCycle)

        # call function
        ind = self.stream.integers(0, length)
        q, qxy = self.propCycle[ind](x, iter, self.beta)

        # axuilary jump
        if len(self.aux) > 0:
            for aux in self.aux:
                q, qxy_aux = aux(x, q, iter, self.beta)
                qxy += qxy_aux

        return q, qxy, self.propCycle[ind].__name__

    # TODO: jump statistics


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
