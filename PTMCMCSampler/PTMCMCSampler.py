import os
import sys
import time
import warnings

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
    jump proposals with the ``addProposalToCycle`` fuction.

    The user can also choose to perform Model-Switch Thermodynamic
    Integration (MSTI), which uses the Parallel Tempering aspect
    of PTMCMC to sample from varying mixtures of two models'
    posteriors.

    @param ndim: number of dimensions in problem
    @param logl: single log-likelihood function or tuple of log-likelihood functions if
    using MSTI
    @param logp: single log prior function (must be normalized for evidence evaluation) or
    tuple of log prior functions if using MSTI
    @param cov: Initial covariance matrix of model parameters for jump proposals
    @param covinds: Indices of parameters for which to perform adaptive jumps
    @param loglargs: any additional arguments (apart from the parameter vector) for
    log likelihood
    @param loglkwargs: any additional keyword arguments (apart from the parameter vector)
    for log likelihood
    @param logpargs: any additional arguments (apart from the parameter vector) for
    log like prior
    @param logl_grad: log-likelihood function, including gradients
    @param logp_grad: prior function, including gradients
    @param logpkwargs: any additional keyword arguments (apart from the parameter vector)
    for log prior
    @param outDir: Full path to output directory for chain files (default = ./chains)
    @param verbose: Update current run-status to the screen (default=True)
    @param resume: Resume from a previous chain (still in testing so beware) (default=False)

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

        # if 2 loglikelihood functions and 2 log prior functions are supplied (MSTI)
        if (type(logl) is tuple) and (type(logp) is tuple):
            self.logl1 = _function_wrapper(logl[1], loglargs, loglkwargs)
            self.logl2 = _function_wrapper(logl[0], loglargs, loglkwargs)
            self.logp1 = _function_wrapper(logp[1], logpargs, logpkwargs)
            self.logp2 = _function_wrapper(logp[0], logpargs, logpkwargs)
        elif (type(logl) is tuple) or (type(logp) is tuple):
            raise ValueError(
                "You provided a tuple for either the loglikelihood or log prior but not the other."
                "If you are using MSTI make sure both are tuples."
            )
        else:  # only 1 loglikelihood and 1 log prior provided
            self.logl = _function_wrapper(logl, loglargs, loglkwargs)
            self.logp = _function_wrapper(logp, logpargs, logpkwargs)

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
        shape="geometric",
        Bmax=1,
        Bmin=None,
        Tmin=None,
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
        MSTI=False,
        nameChainTemps=False,
        model_param_idx=None,
    ):
        """
        Initialize MCMC quantities
        @param p0: Initial parameter vector
        @param self.Niter: Number of iterations to use for T = 1 chain
        @param Bmax: Maximum beta in ladder (default=1)
        @param Bmin: Minimum beta in ladder (default=None)
        @param ladder: User defined temperature/beta ladder. Either scheme accepted.
        @param shape: Specifies shape of beta/temperature ladder if a ladder is not already given
        (default='geometric')
        @param Tmin: Minimum temperature in ladder (default=None)
        @param Tmax: Maximum temperature in ladder (default=None)
        @param Tskip: Number of steps between proposed temperature swaps (default=100)
        @param isave: Write to file every isave samples (default=1000)
        @param covUpdate: Number of iterations between AM covariance updates (default=1000)
        @param SCAMweight: Weight of SCAM jumps in overall jump cycle (default=20)
        @param AMweight: Weight of AM jumps in overall jump cycle (default=20)
        @param DEweight: Weight of DE jumps in overall jump cycle (default=20)
        @param NUTSweight: Weight of the NUTS jumps in jump cycle (default=20)
        @param MALAweight: Weight of the MALA jumps in jump cycle (default=20)
        @param HMCweight: Weight of the HMC jumps in jump cycle (default=20)
        @param HMCstepsize: Step-size of the HMC jumps (default=0.1)
        @param HMCsteps: Maximum number of steps in an HMC trajectory (default=300)
        @param burn: Burn in time (DE jumps added after this iteration) (default=10000)
        @param maxIter: Maximum number of iterations for high temperature chains
                        (default=2*self.Niter)
        @param self.thin: MCMC Samples are recorded every self.thin samples
        @param i0: Iteration to start MCMC (if i0 !=0, do not re-initialize)
        @param neff: Number of effective samples to collect before terminating
        @param writeHotChains: Writes out the hot chain (default=False)
        @param hotChain: Beta=0 (previously Temp=1e80) (default=False)
        @param MSTI: Indicates whether or not to use Model-Switch Thermodynamic Integration
                    (default=False)
        @param model_param_idx: A tuple of lists of indices for each model’s parameters in MSTI,
                        used to sift through the combined set of parameters in p0. Only needed if
                        the two models have different parameters (default=None)
        @param nameChainTemps: Reverts to temperature naming convention of chains (default=False)


        """
        # get maximum number of iteration
        if maxIter is None and self.MPIrank > 0:
            maxIter = Niter
        elif maxIter is None and self.MPIrank == 0:
            maxIter = Niter

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
        self.ind_next_write = 0  # Next index in these arrays to write out
        self.naccepted = 0
        self.swapProposed = 0
        self.nswap_accepted = 0
        self.n_metaparams = 4
        self.MSTI = MSTI
        self.model_param_idx = model_param_idx

        if not self.MSTI:
            if hasattr(self, "logl1"):
                raise ValueError(
                    "You have provided a likelihood and a prior function for two models but have"
                    " indicated that MSTI should not be performed."
                )

        if self.MSTI:
            if hasattr(self, "logl"):
                raise ValueError(
                    "You have indicated that MSTI should be performed but have only provided"
                    " one likelihood and one prior function. For MSTI you must supply one of each"
                    " for each model."
                )
            self.n_metaparams = 8
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

        # if ladder given check if in temp or beta
        if self.ladder:
            if any(self.ladder) > 1:
                # user gave temperatures >>> convert to beta
                self.ladder = [1 / temp for temp in self.ladder]

        # ladder not specified, create one
        else:
            # If temperatures are used, convert to beta
            if Tmin:  # used temperatures
                Bmax = 1 / Tmin  # Tmin is typically 1
            if Tmax:
                Bmin = 1 / Tmax

            self.ladder = self.Ladder(Bmax, Bmin=Bmin, shape=shape)

        # beta for current chain
        self.beta = self.ladder[self.MPIrank]

        # Name chain files
        if hotChain and self.MPIrank == self.nchain - 1:
            self.beta = 0  # This is the "hot chain"
            if nameChainTemps:  # if you prefer the old naming scheme
                self.fname = self.outDir + "/chain_hot.txt"
            else:  # new naming scheme with beta
                self.fname = self.outDir + "/chain_0.txt"

        elif nameChainTemps:  # if you prefer the old naming scheme
            self.fname = self.outDir + "/chain_{0}.txt".format(1 / self.beta)

        else:  # new naming scheme with beta
            self.fname = self.outDir + "/chain_{0}.txt".format(self.beta)

        # write hot chains
        self.writeHotChains = writeHotChains

        self.resumeLength = 0
        if self.resume and os.path.isfile(self.fname):
            if self.verbose:
                print("Resuming run from chain file {0}".format(self.fname))
            try:
                self.resumechain = np.loadtxt(self.fname, ndmin=2)
                self.resumeLength = self.resumechain.shape[0]  # Number of samples read from old chain
            except ValueError as error:
                print("Reading old chain files failed with error", error)
                raise Exception("Couldn't read old chain to resume")
            self._chainfile = open(self.fname, "a")
            if (
                self.isave != self.thin  # This special case is always OK
                and self.resumeLength % (self.isave / self.thin) != 1  # Initial sample plus blocks of isave/thin
            ):
                raise Exception(
                    (
                        "Old chain has {0} rows, which is not the initial sample plus a multiple of isave/thin = {1}"
                    ).format(self.resumeLength, self.isave // self.thin)
                )
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
        Update chains after jump proposals

        """
        # update buffer
        if self.MPIrank == 0:
            self._AMbuffer[iter % self.covUpdate, :] = p0

        # put results into arrays
        if iter % self.thin == 0:
            ind = int(iter / self.thin)
            self._chain[ind, :] = p0
            self._lnlike[ind] = lnlike0
            self._lnprob[ind] = lnprob0

            if lnlike1 and lnlike2 and lnprob1 and lnprob2:
                self._lnlike1[ind] = lnlike1
                self._lnprob1[ind] = lnprob1
                self._lnlike2[ind] = lnlike2
                self._lnprob2[ind] = lnprob2

        # write to file
        if iter % self.isave == 0:
            self.writeOutput(iter)

    def writeOutput(self, iter):
        """
        Write chains and covariance matrix.  Called every isave on samples or at end.
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
        Bmax=1,
        Bmin=None,
        ladder=None,
        shape="geometric",
        Tmin=None,
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
        MSTI=False,
        model_param_idx=None,
        nameChainTemps=False,
    ):
        """
        Function to carry out PTMCMC sampling.

        @param p0: Initial parameter vector
        @param self.Niter: Number of iterations to use for T = 1 chain
        @param Bmax: Maximum beta in ladder (default=1)
        @param Bmin: Minimum beta in ladder (default=None)
        @param shape: Specifies shape of beta/temperature ladder if a ladder is not already
                    given (default='geometric')
        @param ladder: User defined temperature/beta ladder. Either scheme accepted
        @param Tmin: Minimum temperature in ladder (default=None)
        @param Tmax: Maximum temperature in ladder (default=None)
        @param Tskip: Number of steps between proposed temperature swaps (default=100)
        @param isave: Write to file every isave samples (default=1000)
        @param covUpdate: Number of iterations between AM covariance updates (default=1000)
        @param SCAMweight: Weight of SCAM jumps in overall jump cycle (default=20)
        @param AMweight: Weight of AM jumps in overall jump cycle (default=20)
        @param DEweight: Weight of DE jumps in overall jump cycle (default=20)
        @param NUTSweight: Weight of the NUTS jumps in jump cycle (default=20)
        @param MALAweight: Weight of the MALA jumps in jump cycle (default=20)
        @param HMCweight: Weight of the HMC jumps in jump cycle (default=20)
        @param HMCstepsize: Step-size of the HMC jumps (default=0.1)
        @param HMCsteps: Maximum number of steps in an HMC trajectory (default=300)
        @param burn: Burn in time (DE jumps added after this iteration) (default=10000)
        @param maxIter: Maximum number of iterations for high temperature chains
                    (default=2*self.Niter)
        @param self.thin: MCMC Samples are recorded every self.thin samples
        @param i0: Iteration to start MCMC (if i0 !=0, do not re-initialize)
        @param neff: Number of effective samples to collect before terminating
        @param writeHotChains: Writes out the hot chain (default=False)
        @param hotChain: Beta=0 (previously Temp=1e80) (default=False)
        @param MSTI: Indicates whether or not to use Model-Switch Thermodynamic Integration
                    (default=False)
        @param model_param_idx: A tuple of lists of indices for each model’s parameters in MSTI,
                    used to sift through the combined set of parameters in p0. (default=None)
        @param nameChainTemps: Reverts to temperature naming convention of chains (default=False)

        """

        # get maximum number of iteration
        if maxIter is None and self.MPIrank > 0:
            maxIter = Niter
        elif maxIter is None and self.MPIrank == 0:
            maxIter = Niter

        if isave % thin != 0:
            raise ValueError("isave = %d is not a multiple of thin =  %d" % (isave, thin))

        if Niter % thin != 0:
            print(
                "Niter = %d is not a multiple of thin = %d.  The last %d samples will be lost"
                % (Niter, thin, Niter % thin)
            )

        # set up arrays to store lnprob, lnlike and chain
        # if picking up from previous run, don't re-initialize
        if i0 == 0:
            self.initialize(
                Niter,
                Bmax=Bmax,
                Bmin=Bmin,
                ladder=ladder,
                shape=shape,
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
                hotChain=hotChain,
                MSTI=MSTI,
                model_param_idx=model_param_idx,
                nameChainTemps=nameChainTemps,
            )

        # compute lnprob for initial point in chain

        # if resuming, just start with first point in chain
        if self.resume and self.resumeLength > 0:
            p0 = self.resumechain[0, : -self.n_metaparams]
            lnlike0 = self.resumechain[0, -(self.n_metaparams - 1)]
            lnprob0 = self.resumechain[0, -self.n_metaparams]

            if self.MSTI:
                lnlike1 = self.resumechain[0, -(self.n_metaparams - 3)]
                lnprob1 = self.resumechain[0, -(self.n_metaparams - 2)]
                lnlike2 = self.resumechain[0, -(self.n_metaparams - 5)]
                lnprob2 = self.resumechain[0, -(self.n_metaparams - 4)]

            self.ind_next_write = self.resumeLength

        else:
            # compute prior and likelihood
            if not self.MSTI:
                lp = self.logp(p0)

                if lp == -np.inf:
                    lnprob0 = -np.inf

                else:
                    lnlike0 = self.logl(p0)
                    lnprob0 = self.beta * lnlike0 + lp

            elif self.MSTI:  # Using MSTI
                if self.model_param_idx:
                    y1 = [p0[idx] for idx in self.model_param_idx[1]]
                    y2 = [p0[idx] for idx in self.model_param_idx[0]]

                else:
                    y1 = p0
                    y2 = p0

                lp1 = self.logp1(y1)
                lp2 = self.logp2(y2)

                if lp1 == -np.inf or lp2 == -np.inf:
                    lnprob0 = -np.inf

                else:
                    lnlike1 = self.logl1(y1)
                    lnprob1 = lnlike1 + lp1

                    lnlike2 = self.logl2(y2)
                    lnprob2 = lnlike2 + lp2

                    lnlike0 = lnprob1 - lnprob2

                    lnprob0 = self.beta * (lnlike0) + lnprob2

        if not self.MSTI:
            # record first values
            self.updateChains(p0, lnlike0, lnprob0, i0)

        elif MSTI:  # Using MSTI
            # record first values
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
            if not self.MSTI:
                p0, lnlike0, lnprob0 = self.PTMCMCOneStep(p0, lnlike0, lnprob0, iter)
            elif self.MSTI:
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
        Function to carry out PTMCMC sampling.

        @param p0: Initial parameter vector
        @param lnlike0: Initial log-likelihood value
        @param lnprob0: Initial log probability value
        @param iter: iteration number

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

        # jump proposal ###

        # if resuming, just use previous chain points.  Use each one thin times to compensate for
        # thinning when they were written out
        if self.resume and self.resumeLength > 0 and iter < self.resumeLength * self.thin:
            p0 = self.resumechain[0, : -self.n_metaparams]
            lnlike0 = self.resumechain[0, -(self.n_metaparams - 1)]
            lnprob0 = self.resumechain[0, -self.n_metaparams]

            if self.MSTI:
                lnlike1 = self.resumechain[0, -(self.n_metaparams - 3)]
                lnprob1 = self.resumechain[0, -(self.n_metaparams - 2)]
                lnlike2 = self.resumechain[0, -(self.n_metaparams - 5)]
                lnprob2 = self.resumechain[0, -(self.n_metaparams - 4)]

            # update acceptance counter
            self.naccepted = iter * self.resumechain[iter // self.thin, -2]
        else:
            y, qxy, jump_name = self._jump(p0, iter)  # made a jump
            self.jumpDict[jump_name][0] += 1

            # compute prior and likelihood
            if not self.MSTI:
                lp = self.logp(y)

                if lp == -np.inf:
                    newlnprob = -np.inf

                else:
                    newlnlike = self.logl(y)
                    newlnprob = self.beta * newlnlike + lp

            elif self.MSTI:  # Using MSTI

                if self.model_param_idx:
                    y1 = [y[idx] for idx in self.model_param_idx[1]]
                    y2 = [y[idx] for idx in self.model_param_idx[0]]

                else:
                    y1 = y
                    y2 = y

                lp1 = self.logp1(y1)
                lp2 = self.logp2(y2)

                if lp1 == -np.inf or lp2 == -np.inf:
                    newlnprob = -np.inf

                else:
                    newlnlike1 = self.logl1(y1)
                    newlnprob1 = newlnlike1 + lp1  # no beta here, we want full posterior

                    newlnlike2 = self.logl2(y2)
                    newlnprob2 = newlnlike2 + lp2  # no beta here, we want full posterior

                    newlnlike = newlnprob1 - newlnprob2

                    # ln posterior = beta * ln likelihood + ln prior
                    # ln prior is set to ln posterior of the second model
                    # ln likelihood is the difference between ln posterior of the first and second models
                    # beta determines how much of newlnprob1 vs newlnprob2
                    newlnprob = self.beta * (newlnlike) + newlnprob2

            # hastings step
            diff = newlnprob - lnprob0 + qxy

            rand_log = np.log(self.stream.random())
            if diff > rand_log:

                # accept jump
                p0, lnlike0, lnprob0 = y, newlnlike, newlnprob
                if self.MSTI:
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
        if self.MSTI:
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
                p0, lnlike0, lnprob0 = self.PTswap(p0, lnlike0, lnprob0, iter)  # No temperature swap for MSTI

            self.updateChains(p0, lnlike0, lnprob0, iter)

            return p0, lnlike0, lnprob0

    def PTswap(self, p0, lnlike0, lnprob0, iter):
        """
        Do parallel tempering swap. This feature is not compatible with
        Model-Switch Thermodynamic Integraton (MSTI)

        (Repurposed from Neil Cornish/Bence Becsy's code)

        Swap acceptance rates are computed per chain by storing
        the number of swaps proposed and accepted. Since swaps
        are proposed for every chain, swapProposed is always
        incremented and nswap_accepted will be incremented only
        for chains that have the swap accepted. The swap acceptance
        is calculated elsewhere.

        @param p0: current parameter vector
        @param lnlike0: current log-likelihood
        @param lnprob0: current log posterior value
        @param iter: current iteration number

        @return swapReturn: 0 = no swap proposed,
        1 = swap proposed and rejected,
        2 = swap proposed and accepted

        @return p0: new parameter vector
        @return lnlike0: new log-likelihood
        @return lnprob0: new log posterior value

        """
        Ts = self.ladder  # as beta

        log_Ls = self.comm.gather(lnlike0, root=0)  # list of likelihoods from each chain
        p0s = self.comm.gather(p0, root=0)  # list of parameter arrays from each chain

        if self.MPIrank == 0:
            # set up map to help keep track of swaps
            swap_map = list(range(self.nchain))

            # loop through and propose a swap at each chain (starting from hottest chain and going down in T)
            # and keep track of results in swap_map
            for swap_chain in reversed(range(self.nchain - 1)):

                log_acc_ratio = -log_Ls[swap_map[swap_chain]] * Ts[swap_chain]
                log_acc_ratio += -log_Ls[swap_map[swap_chain + 1]] * Ts[swap_chain + 1]
                log_acc_ratio += log_Ls[swap_map[swap_chain + 1]] * Ts[swap_chain]
                log_acc_ratio += log_Ls[swap_map[swap_chain]] * Ts[swap_chain + 1]

                acc_ratio = np.exp(log_acc_ratio)

                if self.stream.uniform() <= acc_ratio:
                    swap_map[swap_chain], swap_map[swap_chain + 1] = (
                        swap_map[swap_chain + 1],
                        swap_map[swap_chain],
                    )
                    self.nswap_accepted += 1
                    self.swapProposed += 1
                else:
                    self.swapProposed += 1

            # loop through the chains and record the new samples and log_Ls
            for j in range(self.nchain):
                p0s[j] = p0s[swap_map[j]]
                log_Ls[j] = log_Ls[swap_map[j]]

        # broadcast the new samples and log_Ls to all chains
        p0 = self.comm.scatter(p0s, root=0)
        lnlike0 = self.comm.scatter(log_Ls, root=0)

        # calculate new posterior values
        lnprob0 = self.beta * lnlike0 + self.logp(p0)

        return p0, lnlike0, lnprob0

    def Ladder(self, Bmax, Bmin=None, tstep=None, shape="geometric"):
        """
        Method to compute temperature/beta ladder. The default is a geometrically
        spaced ladder with a spacing designed to give 25 % temperature/beta swap
        acceptance rate. The other option is a linear spacing.

        """

        # TODO: make options to do other temperature ladders

        if self.nchain > 1:
            if shape == "linear":
                if tstep is None and Bmin is None:  # Bmin set to 0
                    if Bmin is None:
                        warnings.warn("Bmin not given. Bmin will be set to 0 for linear spacing.")
                        Bmin = 0
                    tstep = Bmax / (self.nchain - 1)

                elif tstep is None and Bmin is not None:
                    tstep = (Bmax - Bmin) / (self.nchain - 1)

                ladder = np.zeros(self.nchain)
                for ii in range(self.nchain):
                    ladder[ii] = round(Bmax - (tstep * ii), 5)

            if shape == "geometric":
                if tstep is None and Bmin is None:
                    tstep = 1 + np.sqrt(2 / self.ndim)

                elif tstep is None and Bmin is not None:
                    if Bmin == 0:
                        warnings.warn(
                            "Bmin set to 0. Geometric series can only approach beta=0. Make sure to include the"
                            "hot chain to get a beta=0 chain if you haven't already. Bmin will be set to 1e-7."
                        )
                        Bmin = 1e-7
                    tstep = np.exp(np.log(Bmin / Bmax) / (1 - self.nchain))  # Bmin can't be 0 here

                ladder = np.zeros(self.nchain)
                for ii in range(self.nchain):
                    ladder[ii] = Bmax * tstep ** (-ii)
        else:
            ladder = np.array([Bmax])

        return ladder

    def _writeToFile(self, iter):
        """
        Function to write chain file. File has ndim+4 columns,
        appended to the parameter values are log-posterior (unnormalized),
        log-likelihood, acceptance rate, and PT acceptance rate. If doing MSTI
        there are an additional 4 colums (ndim+8 total), log-posterior of model
        1, log-likelihood of model 1, log-posterior of model 2, and
        log-likelihood of model 2.
        Rates are as of time of writing.

        @param iter: Iteration of sampler

        """

        self._chainfile = open(self.fname, "a+")
        # index 0 is the initial element.  So after 10*thin iterations we need to write elements 1..10
        write_end = iter // self.thin + 1  # First element not to write.
        for ind in range(self.ind_next_write, write_end):
            pt_acc = 1
            if self.MPIrank < self.nchain - 1 and self.swapProposed != 0:
                pt_acc = self.nswap_accepted / self.swapProposed

            self._chainfile.write("\t".join(["%22.22f" % (self._chain[ind, kk]) for kk in range(self.ndim)]))
            self._chainfile.write("\t%f\t%f" % (self._lnprob[ind], self._lnlike[ind]))

            if self.MSTI:  # MSTI
                self._chainfile.write(
                    "\t%f\t%f\t%f\t%f"
                    % (
                        self._lnprob1[ind],
                        self._lnlike1[ind],
                        self._lnprob2[ind],
                        self._lnlike2[ind],
                    )
                )

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
        Update Differential Evolution with last burn
        values in the total chain

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

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

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

        # adjust scale based on beta
        # if self.beta >= 0.01:
        #     scale *= 1/np.sqrt(self.beta)

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

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

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

        # adjust scale based on beta
        # if self.beta >= 0.01:
        #     scale *= 1/np.sqrt(self.beta)

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
        Differential Evolution Jump. This function will  occasionally
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

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
        # prob = self.stream.random()

        # mode jump
        # if prob > 0.5:
        #     scale = 1.0

        # else:
        #     scale = self.stream.random() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1 / self.beta)

        scale = 1.0

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
        pulsar phase fixes

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
        Call Jump proposals

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
