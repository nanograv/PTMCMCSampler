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
    print(
        "Optional acor package is not installed. Acor is optionally used to calculate the "
        "effective chain length for output in the chain file."
    )
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

    @param ndim: number of dimensions in problem
    @param logl: log-likelihood function
    @param logp: log prior function (must be normalized for evidence evaluation)
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
        neff=100000,
        writeHotChains=False,
        hotChain=False,
    ):
        """
        Initialize MCMC quantities

        @param maxIter: maximum number of iterations
        @Tmin: minumum temperature to use in temperature ladder

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

        N = int(maxIter / thin)

        self._lnprob = np.zeros(N)
        self._lnlike = np.zeros(N)
        self._chain = np.zeros((N, self.ndim))
        self.naccepted = 0
        self.swapProposed = 0
        self.nswap_accepted = 0

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

        # setup default temperature ladder
        if self.ladder is None:
            self.ladder = self.temperatureLadder(Tmin, Tmax=Tmax)

        # temperature for current chain
        self.temp = self.ladder[self.MPIrank]

        # hot chain sampling from prior
        if hotChain and self.MPIrank == self.nchain - 1:
            self.temp = 1e80
            self.fname = self.outDir + "/chain_hot.txt"
        else:
            self.fname = self.outDir + "/chain_{0}.txt".format(self.temp)

        # write hot chains
        self.writeHotChains = writeHotChains

        self.resumeLength = 0
        if self.resume and os.path.isfile(self.fname):
            if self.verbose:
                print("Resuming run from chain file {0}".format(self.fname))
            try:
                self.resumechain = np.loadtxt(self.fname)
                self.resumeLength = self.resumechain.shape[0]
            except ValueError:
                print("WARNING: Cant read in file. Removing last line.")
                os.system("sed -ie '$d' {0}".format(self.fname))
                self.resumechain = np.loadtxt(self.fname)
                self.resumeLength = self.resumechain.shape[0]
            self._chainfile = open(self.fname, "a")
        else:
            self._chainfile = open(self.fname, "w")
        self._chainfile.close()

    def updateChains(self, p0, lnlike0, lnprob0, iter):
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

        # write to file
        if iter % self.isave == 0 and iter > 1 and iter > self.resumeLength:
            if self.writeHotChains or self.MPIrank == 0:
                self._writeToFile(iter)

            # write output covariance matrix
            np.save(self.outDir + "/cov.npy", self.cov)
            if self.MPIrank == 0 and self.verbose and iter > 1:
                sys.stdout.write("\r")
                sys.stdout.write(
                    "Finished %2.2f percent in %f s Acceptance rate = %g"
                    % (iter / self.Niter * 100, time.time() - self.tstart, self.naccepted / iter)
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
        neff=100000,
        writeHotChains=False,
        hotChain=False,
    ):
        """
        Function to carry out PTMCMC sampling.

        @param p0: Initial parameter vector
        @param self.Niter: Number of iterations to use for T = 1 chain
        @param ladder: User defined temperature ladder
        @param Tmin: Minimum temperature in ladder (default=1)
        @param Tmax: Maximum temperature in ladder (default=None)
        @param Tskip: Number of steps between proposed temperature swaps (default=100)
        @param isave: Number of iterations before writing to file (default=1000)
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
        @param self.thin: Save every self.thin MCMC samples
        @param i0: Iteration to start MCMC (if i0 !=0, do not re-initialize)
        @param neff: Number of effective samples to collect before terminating

        """

        # get maximum number of iteration
        if maxIter is None and self.MPIrank > 0:
            maxIter = Niter
        elif maxIter is None and self.MPIrank == 0:
            maxIter = Niter

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
                hotChain=hotChain,
            )

        # compute lnprob for initial point in chain

        # if resuming, just start with first point in chain
        if self.resume and self.resumeLength > 0:
            p0, lnlike0, lnprob0 = self.resumechain[0, :-4], self.resumechain[0, -3], self.resumechain[0, -4]
        else:
            # compute prior
            lp = self.logp(p0)

            if lp == float(-np.inf):

                lnprob0 = -np.inf
                lnlike0 = -np.inf

            else:

                lnlike0 = self.logl(p0)
                lnprob0 = 1 / self.temp * lnlike0 + lp

        # record first values
        self.updateChains(p0, lnlike0, lnprob0, i0)

        self.comm.barrier()

        # start iterations
        iter = i0
        self.tstart = time.time()
        runComplete = False
        Neff = 0
        while runComplete is False:
            iter += 1
            self.comm.barrier()  # make sure all processes are at the same iteration
            # call PTMCMCOneStep
            p0, lnlike0, lnprob0 = self.PTMCMCOneStep(p0, lnlike0, lnprob0, iter)

            # compute effective number of samples
            if iter % 1000 == 0 and iter > 2 * self.burn and self.MPIrank == 0:
                try:
                    Neff = iter / max(
                        1,
                        np.nanmax([acor.acor(self._chain[self.burn : (iter - 1), ii])[0] for ii in range(self.ndim)]),
                    )
                    # print('\n {0} effective samples'.format(Neff))
                except NameError:
                    Neff = 0
                    pass

            # stop if reached maximum number of iterations
            if self.MPIrank == 0 and iter >= self.Niter - 1:
                if self.verbose:
                    print("\nRun Complete")
                runComplete = True

            # stop if reached effective number of samples
            if self.MPIrank == 0 and int(Neff) > self.neff:
                if self.verbose:
                    print("\nRun Complete with {0} effective samples".format(int(Neff)))
                runComplete = True

            runComplete = self.comm.bcast(runComplete, root=0)

    def PTMCMCOneStep(self, p0, lnlike0, lnprob0, iter):
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

        # if resuming, just use previous chain points
        if self.resume and self.resumeLength > 0 and iter < self.resumeLength:
            p0, lnlike0, lnprob0 = self.resumechain[iter, :-4], self.resumechain[iter, -3], self.resumechain[iter, -4]

            # update acceptance counter
            self.naccepted = iter * self.resumechain[iter, -2]
        else:
            y, qxy, jump_name = self._jump(p0, iter)
            self.jumpDict[jump_name][0] += 1

            # compute prior and likelihood
            lp = self.logp(y)

            if lp == -np.inf:

                newlnprob = -np.inf

            else:

                newlnlike = self.logl(y)
                newlnprob = 1 / self.temp * newlnlike + lp

            # hastings step
            diff = newlnprob - lnprob0 + qxy
            if diff > np.log(self.stream.random()):

                # accept jump
                p0, lnlike0, lnprob0 = y, newlnlike, newlnprob

                # update acceptance counter
                self.naccepted += 1
                self.jumpDict[jump_name][1] += 1
        # temperature swap
        if iter % self.Tskip == 0 and self.nchain > 1:
            p0, lnlike0, lnprob0 = self.PTswap(p0, lnlike0, lnprob0, iter)

        self.updateChains(p0, lnlike0, lnprob0, iter)

        return p0, lnlike0, lnprob0

    def PTswap(self, p0, lnlike0, lnprob0, iter):
        """
        Do parallel tempering swap.

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

        Repurposed from Neil Cornish/Bence Becsy's code:
        """
        Ts = self.ladder

        log_Ls = self.comm.gather(lnlike0, root=0)  # list of likelihoods from each chain
        p0s = self.comm.gather(p0, root=0)  # list of parameter arrays from each chain

        if self.MPIrank == 0:
            # set up map to help keep track of swaps
            swap_map = list(range(self.nchain))

            # loop through and propose a swap at each chain (starting from hottest chain and going down in T)
            # and keep track of results in swap_map
            for swap_chain in reversed(range(self.nchain - 1)):
                log_acc_ratio = -log_Ls[swap_map[swap_chain]] / Ts[swap_chain]
                log_acc_ratio += -log_Ls[swap_map[swap_chain + 1]] / Ts[swap_chain + 1]
                log_acc_ratio += log_Ls[swap_map[swap_chain + 1]] / Ts[swap_chain]
                log_acc_ratio += log_Ls[swap_map[swap_chain]] / Ts[swap_chain + 1]

                acc_ratio = np.exp(log_acc_ratio)
                if self.stream.uniform() <= acc_ratio:
                    swap_map[swap_chain], swap_map[swap_chain + 1] = swap_map[swap_chain + 1], swap_map[swap_chain]
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
        lnprob0 = 1 / self.temp * lnlike0 + self.logp(p0)

        return p0, lnlike0, lnprob0

    def temperatureLadder(self, Tmin, Tmax=None, tstep=None):
        """
        Method to compute temperature ladder. At the moment this uses
        a geometrically spaced temperature ladder with a temperature
        spacing designed to give 25 % temperature swap acceptance rate.

        """

        # TODO: make options to do other temperature ladders

        if self.nchain > 1:
            if tstep is None and Tmax is None:
                tstep = 1 + np.sqrt(2 / self.ndim)
            elif tstep is None and Tmax is not None:
                tstep = np.exp(np.log(Tmax / Tmin) / (self.nchain - 1))
            ladder = np.zeros(self.nchain)
            for ii in range(self.nchain):
                ladder[ii] = Tmin * tstep**ii
        else:
            ladder = np.array([1])

        return ladder

    def _writeToFile(self, iter):
        """
        Function to write chain file. File has 3+ndim columns,
        the first is log-posterior (unweighted), log-likelihood,
        and acceptance probability, followed by parameter values.

        @param iter: Iteration of sampler

        """

        self._chainfile = open(self.fname, "a+")
        for jj in range((iter - self.isave), iter, self.thin):
            ind = int(jj / self.thin)
            pt_acc = 1
            if self.MPIrank < self.nchain - 1 and self.swapProposed != 0:
                pt_acc = self.nswap_accepted / self.swapProposed

            self._chainfile.write("\t".join(["%22.22f" % (self._chain[ind, kk]) for kk in range(self.ndim)]))
            self._chainfile.write(
                "\t%f\t%f\t%f\t%f\n" % (self._lnprob[ind], self._lnlike[ind], self.naccepted / iter, pt_acc)
            )
        self._chainfile.close()

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

        # standard medium jump
        else:
            scale = 1.0

        # adjust scale based on temperature
        if self.temp <= 100:
            scale *= np.sqrt(self.temp)

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

        # adjust scale based on temperature
        if self.temp <= 100:
            scale *= np.sqrt(self.temp)

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
        prob = self.stream.random()

        # mode jump
        if prob > 0.5:
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
        q, qxy = self.propCycle[ind](x, iter, 1 / self.temp)

        # axuilary jump
        if len(self.aux) > 0:
            for aux in self.aux:
                q, qxy_aux = aux(x, q, iter, 1 / self.temp)
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
