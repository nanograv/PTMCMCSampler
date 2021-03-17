"""
Implementation of the No-U-Turn-Sampler. Code follows algorithm 6 from the NUTS
paper (Hoffman & Gelman, 2011)

reference: arXiv:1111.4246
"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
Carlo", Matthew D. Hoffman & Andrew Gelman

Rutger van Haasteren
"""


from __future__ import division, print_function

import os
import sys

import numpy as np
import scipy.linalg as sl


class GradientJump(object):
    """Class for jumps using gradient information"""

    def __init__(self, loglik_grad, logprior_grad, mm_inv, nburn=100):
        """Initialize the HMC class

        :loglik_grad:   Log-likelihood and gradient function
        :logprior_grad: Log-prior and gradient function
        :mm_inv:        Inverse of the mass matrix (covariance matrix)
        :nburn:         Number of burn-in steps
        """

        self._loglik_grad = loglik_grad  # Log-likelihood & gradient
        self._logprior_grad = logprior_grad  # Log-prior & gradient
        self.mm_inv = mm_inv  # Inverse mass-matrix
        self.nburn = nburn  # Nr. of burn-in steps
        self.ndim = len(self.mm_inv)  # Number of dimensions
        self.set_cf()  # Whitening matrices

        self.name = "GradientJUMP"

        self.epsilon = None  # Step-size
        self.beta = 1.0  # Inverse temperature
        self.iter = 0.0  # Number of gradient jumps

        print("WARNING: GradientJumps not yet adaptive. Choose cov wisely!")

    @property
    def __name__(self):
        return self.name

    def set_cf(self):
        """Update the Cholesky factor of the inverse mass matrix"""
        self.cov_cf = sl.cholesky(self.mm_inv, lower=True)
        self.cov_cfi = sl.solve_triangular(self.cov_cf, np.eye(len(self.cov_cf)), trans=0, lower=True)

    def update_cf(self):
        """Update the Cholesky factor of the inverse mass matrix

        NOTE: this function is different from the one in GradientJump!
        """
        # Since we are adaptively tuning the step size epsilon, we should at
        # least keep the determinant of this guy equal to what it was before.
        new_cov_cf = sl.cholesky(self.mm_inv, lower=True)

        ldet_old = np.sum(np.log(np.diag(self.cov_cf)))
        ldet_new = np.sum(np.log(np.diag(new_cov_cf)))

        self.cov_cf = np.exp((ldet_old - ldet_new) / self.ndim) * new_cov_cf
        self.cov_cfi = sl.solve_triangular(self.cov_cf, np.eye(len(self.cov_cf)), trans=0, lower=True)

    def func_grad(self, x):
        """Log-prob and gradient, corrected for temperature"""
        ll, ll_grad = self._loglik_grad(x)
        lp, lp_grad = self._logprior_grad(x)

        return self.beta * ll + lp, self.beta * ll_grad + lp_grad

    def forward(self, x):
        """Coordinate transformation to whitened parameters x->q"""
        return np.dot(self.cov_cfi.T, x)

    def backward(self, q):
        """Coordinate transformation from whitened parameters q->x"""
        return np.dot(self.cov_cf.T, q)

    def func_grad_white(self, q):
        """Whitened version of func_grad"""
        x = self.backward(q)
        fv, fg = self.func_grad(x)
        return fv, np.dot(self.cov_cf, fg)

    def draw_momenta(self):
        """Draw new momentum variables"""
        return np.random.randn(len(self.mm_inv))

    def loghamiltonian(self, logl, r):
        """Value of the Hamiltonian, given a position and momentum value"""
        try:
            return logl - 0.5 * np.dot(r, r)
        except ValueError:
            return np.nan

    def posmom_inprod(self, theta, r):
        try:
            return np.dot(theta, r)
        except ValueError:
            return np.nan

    # With the following definitions, we should be able to get rid of this
    # awkward coordinate transformation. Why does it not work?
    """
    def forward(self, x):
        #return np.dot(self.cov_cfi.T, x)
        return x

    def backward(self, q):
        #return np.dot(self.cov_cf.T, q)
        return q

    def func_grad_white(self, q):
        # We would be able to get rid of this function
        #x = self.backward(q)
        #fv, fg = self.func_grad(x)
        #return fv, np.dot(self.cov_cf, fg)
        return self.func_grad(q)

    def draw_momenta(self):
        #return np.random.randn(len(self.mm_inv))
        return np.dot(self.cov_cfi.T, np.random.randn(self.ndim))

    def loghamiltonian(self, logl, r):
        try:
            #return logl-0.5*np.dot(r, r)
            newr = np.dot(self.cov_cfi.T, r)
            return logl-0.5*np.dot(newr, newr)
        except ValueError as err:
            return np.nan

    def posmom_inprod(self, theta, r):
        try:
            newr = np.dot(self.cov_cfi.T, r)
            newtheta = np.dot(self.cov_cfi.T, theta)
            return np.dot(newtheta, newr)
            #return np.dot(theta, r)
        except ValueError as err:
            return np.nan
    """

    def leapfrog(self, theta, r, grad, epsilon):
        """Perfom a leapfrog jump in the Hamiltonian space

        :theta:     Initial parameter position
        :r:         Initial momentum
        :grad:      Initial gradient
        :epsilon:   Step size

        output
        thetaprime: new parameter position
        rprime:     new momentum
        gradprime:  new gradient
        logpprime:  new log-probability
        """

        rprime = r + 0.5 * epsilon * grad  # half step in r
        thetaprime = theta + epsilon * rprime  # step in theta
        logpprime, gradprime = self.func_grad_white(thetaprime)  # compute gradient
        rprime = rprime + 0.5 * epsilon * gradprime  # half step in r

        return thetaprime, rprime, gradprime, logpprime

    def __call__(self, x, iter, beta):
        """Take one HMC trajectory step"""

        self.iter += 1

        if self.__name__ == "GradientJUMP":
            raise NotImplementedError("GradientJump is an abstract base class!")
        else:
            return x, 0.0


class MALAJump(GradientJump):
    """MALA Jump"""

    def __init__(self, loglik_grad, logprior_grad, mm_inv, nburn=100):
        """Initialize the MALA Jump"""
        super(MALAJump, self).__init__(loglik_grad, logprior_grad, mm_inv, nburn=nburn)

        self.name = "MALAJump"
        self.cd = 2.4 / np.sqrt(self.ndim)
        self.set_eigvecs()

    def set_eigvecs(self):
        """Set the eigenvectors of the mass matrix"""
        # Since we have whitened the parameter space, the decomposition is a
        # simple identity matrix
        self._u = np.eye(self.ndim)
        self._s = np.ones(self.ndim)

    def __call__(self, x, iter, beta):
        """Take one MALA step"""
        super(MALAJump, self).__call__(x, iter, beta)

        x = np.atleast_1d(x)
        if len(np.shape(x)) > 1:
            raise ValueError("x is expected to be a 1-D array")

        self.beta = beta

        # Update the mass matrix when in burn-in stage
        # Currently, there is a problem with adjusting the mass matrix.
        # Stepsize and mass matrix need to be tuned together?
        # if iter <= self.nburn:
        #    self.update_cf()
        #    self.set_eigvecs()

        # Initial starting position
        q0 = self.forward(x)
        logp, grad0 = self.func_grad_white(q0)

        # Choose an eigenvector to jump in, and the size
        i = np.random.randint(0, self.ndim)
        vec = self._u[i, :]
        val = self._s[i]
        dist = np.random.randn()

        # Do the leapfrog
        mq0 = q0 + 0.5 * vec * self.cd ** 2 * np.dot(vec, grad0) / 2 / val
        q1 = mq0 + dist * vec * self.cd / np.sqrt(val)
        logp1, grad1 = self.func_grad_white(q1)
        mq1 = q1 + 0.5 * vec * self.cd ** 2 * np.dot(vec, grad1) / 2 / val

        qxy = 0.5 * (np.sum((mq0 - q1) ** 2 / val) - np.sum((mq1 - q0) ** 2 / val))

        return self.backward(q1), qxy


class HMCJump(GradientJump):
    """Hamiltonian Monte Carlo Jump"""

    def __init__(self, loglik_grad, logprior_grad, mm_inv, nburn=100, stepsize=0.1, nminsteps=10, nmaxsteps=300):
        """Initialize the MALA Jump"""
        super(HMCJump, self).__init__(loglik_grad, logprior_grad, mm_inv, nburn=nburn)

        self.name = "HMCJump"
        self.epsilon = stepsize
        self.nminsteps = nminsteps
        self.nmaxsteps = nmaxsteps

    def __call__(self, x, iter, beta):
        """Take one HMC step"""
        super(HMCJump, self).__call__(x, iter, beta)

        x = np.atleast_1d(x)
        if len(np.shape(x)) > 1:
            raise ValueError("x is expected to be a 1-D array")

        # Set the temperature
        self.beta = beta

        # Update the mass matrix when in burn-in stage
        # Currently, there is a problem with adjusting the mass matrix.
        # Stepsize and mass matrix need to be tuned together?
        # if iter <= self.nburn:
        #    self.update_cf()

        # Initial starting position
        q0 = self.forward(x)
        qxy = 0
        logp0, grad0 = self.func_grad_white(q0)

        # Draw new momentum variables
        p0 = self.draw_momenta()
        joint0 = self.loghamiltonian(logp0, p0)

        # Initialize the state
        nsteps = np.random.randint(self.nminsteps, self.nmaxsteps)
        p, q, grad = np.copy(p0), np.copy(q0), np.copy(grad0)

        for ii in range(nsteps):
            q1, p1, grad1, logp1 = self.leapfrog(q, p, grad, self.epsilon)
            joint1 = self.loghamiltonian(logp1, p1)
            p, q, grad = np.copy(p1), np.copy(q1), np.copy(grad1)

            if (joint1 - 1000.0) < joint0:
                # We are super inaccurate, so break the trajectory
                break

        qxy = joint1 - joint0

        return self.backward(q), qxy


class Trajectory(object):
    """Keep track of trajectories in the NUTS jump"""

    def __init__(self, ndim, bufsize=1000):
        """Initialize the trajectory object"""
        self.ndim = ndim
        self.bufadd = bufsize
        self.bufsize_plus = bufsize
        self.bufsize_minus = bufsize
        self.trajlen_plus = 0
        self.trajlen_minus = 0

        self.trajbuf_plus = np.zeros((self.bufsize_plus, self.ndim))
        self.trajind_plus = np.zeros(self.bufsize_plus)
        self.trajbuf_minus = np.zeros((self.bufsize_minus, self.ndim))
        self.trajind_minus = np.zeros(self.bufsize_minus)

    def increase_buf(self, which="plus"):
        """Increase the buffer on the positive or the negative side"""
        addbuf = np.zeros((self.bufadd, self.ndim))
        addind = np.zeros(self.bufadd)

        if which == "plus":
            self.trajbuf_plus = np.append(self.trajbuf_plus, addbuf, axis=0)
            self.trajind_plus = np.append(self.trajind_plus, addind)
            self.bufsize_plus += self.bufadd
        elif which == "minus":
            self.trajbuf_minus = np.append(self.trajbuf_minus, addbuf, axis=0)
            self.trajind_minus = np.append(self.trajind_minus, addind)
            self.bufsize_minus += self.bufadd

    def reset(self):
        """Reset the trajectory object"""
        self.trajlen_plus = 0
        self.trajlen_minus = 0

    def add_sample(self, theta, ind, which="plus"):
        """Add a sample on the positive or the negative branch"""
        if which == "plus":
            if self.trajlen_plus >= self.bufsize_plus:
                self.increase_buf(which="plus")

            self.trajbuf_plus[self.trajlen_plus, :] = theta
            self.trajind_plus[self.trajlen_plus] = ind
            self.trajlen_plus += 1
        elif which == "minus":
            if self.trajlen_minus >= self.bufsize_minus:
                self.increase_buf(which="minus")

            self.trajbuf_minus[self.trajlen_minus, :] = theta
            self.trajind_minus[self.trajlen_minus] = ind
            self.trajlen_minus += 1

    def length(self):
        """Function that returns the current trajectory length"""
        return self.trajlen_plus + self.trajlen_minus

    def get_trajectory(self, which="both"):
        if which == "both":
            return (
                np.append(
                    self.trajbuf_minus[: self.trajlen_minus : -1, :], self.trajbuf_plus[: self.trajlen_plus, :], axis=0
                ),
                np.append(self.trajind_minus[: self.trajlen_minus : -1], self.trajind_plus[: self.trajlen_plus]),
            )
        elif which == "plus":
            return self.trajbuf_plus[: self.trajlen_plus], self.trajind_plus[: self.trajlen_plus]
        elif which == "minus":
            return self.trajbuf_minus[: self.trajlen_minus], self.trajind_minus[: self.trajlen_minus]

    def get_used_trajectory(self, ind):
        """For index ind, get the trajectory that gets us there"""
        tiplus = self.trajind_plus[: self.trajlen_plus]
        timinus = self.trajind_minus[: self.trajlen_minus]

        if ind in tiplus:
            index = np.where(tiplus == ind)[0][0] + 1
            return self.trajbuf_plus[:index, :]
        elif ind in timinus:
            index = np.where(timinus == ind)[0][0] + 1
            return np.append(self.trajbuf_plus[:1, :], self.trajbuf_minus[:index, :], axis=0)
        else:
            raise ValueError("Index not found")


class NUTSJump(GradientJump):
    """Class for No-U-Turn-Sampling Hamiltonian Monte Carlo jumps"""

    def __init__(
        self,
        loglik_grad,
        logprior_grad,
        mm_inv,
        nburn=100,
        trajectoryDir=None,
        write_burnin=False,
        force_trajlen=None,
        force_epsilon=None,
        delta=0.6,
    ):
        """Initialize the HMC class

        :loglik_grad:   Log-likelihood and gradient function
        :logprior_grad: Log-prior and gradient function
        :mm_inv:        Inverse of the mass matrix (covariance matrix)
        :nburn:         Number of burn-in steps
        :trajectoryDir: Output directory for full trajectories (for debugging)
        :write_burnin:  Whether we are writing the burn-in trajectories
        """
        super(NUTSJump, self).__init__(loglik_grad, logprior_grad, mm_inv, nburn=nburn)

        self.trajectoryDir = trajectoryDir  # Trajectory directory
        self.write_burnin = write_burnin  # Write burnin trajectories?

        self.name = "NUTSJUMP"

        self.delta = delta  # Target acceptance rate

        self.traj = Trajectory(self.ndim, bufsize=1000)  # Trajectory buffer

        # Parameters for the dual averaging (for tuning epsilon)
        self.gamma = 0.05
        self.t0 = 10
        self.kappa = 0.75
        self.mu = None
        self.epsilonbar = 1.0
        self.Hbar = 0

        # Parameters to force the trajectories to be pre-set
        self.force_trajlen = force_trajlen
        self.force_epsilon = force_epsilon
        if self.force_epsilon is not None:
            self.epsilonbar = self.force_epsilon

        # Create the trajectory directory, if it does not exist yet
        if self.trajectoryDir is not None:
            if os.path.isfile(trajectoryDir):
                raise IOError("Not a directory: {0}".format(trajectoryDir))
            elif not os.path.isdir(trajectoryDir):
                os.mkdir(trajectoryDir)

    def find_reasonable_epsilon(self, theta0, grad0, logp0):
        """Heuristic for choosing an initial value of epsilon"""

        epsilon = 1.0
        r0 = self.draw_momenta()

        # Figure out what direction we should be moving epsilon.
        _, rprime, gradprime, logpprime = self.leapfrog(theta0, r0, grad0, epsilon)

        # Make sure the step is not too large, so that the likelihood is finite
        # and also stays inside the prior domain (if any)
        k = 1.0
        while np.isinf(logpprime) or np.isinf(gradprime).any():
            k *= 0.5
            _, rprime, _, logpprime = self.leapfrog(theta0, r0, grad0, epsilon * k)

        epsilon = 0.5 * k * epsilon

        acceptprob = np.exp(self.loghamiltonian(logpprime, rprime) - self.loghamiltonian(logp0, r0))

        a = 2.0 * float((acceptprob > 0.5)) - 1.0
        # Keep moving epsilon in that direction until acceptprob crosses 0.5.
        while (acceptprob ** a) > (2.0 ** (-a)):
            epsilon = epsilon * (2.0 ** a)
            _, rprime, _, logpprime = self.leapfrog(theta0, r0, grad0, epsilon)

            acceptprob = np.exp(self.loghamiltonian(logpprime, rprime) - self.loghamiltonian(logp0, r0))

        return epsilon

    def stop_criterion(self, thetaminus, thetaplus, rminus, rplus, force_trajlen, index):
        """ Compute the stop condition in the main loop
        dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

        INPUTS
        ------
        thetaminus, thetaplus: ndarray[float, ndim=1]
            under and above position
        rminus, rplus: ndarray[float, ndim=1]
            under and above momentum

        OUTPUTS
        -------
        criterion: bool
            return if the condition is valid
        """
        dtheta = thetaplus - thetaminus
        inprod_min = self.posmom_inprod(dtheta, rminus)
        inprod_plus = self.posmom_inprod(dtheta, rplus)

        # orig = (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)
        orig = (inprod_min >= 0) & (inprod_plus >= 0)

        if force_trajlen is not None:
            cont = index < force_trajlen
        else:
            cont = orig

        return cont

    def build_tree(self, theta, r, grad, logu, v, j, epsilon, joint0, ind, traj, force_trajlen):
        """The main recursion tree. Literally from Hoffman and Gelman (2011)."""

        if j == 0:
            # Base case: Take a single leapfrog step in the direction v.
            thetaprime, rprime, gradprime, logpprime = self.leapfrog(theta, r, grad, v * epsilon)
            joint = self.loghamiltonian(logpprime, rprime)

            # Is the new point in the slice of the slice sampling step?
            nprime = int(logu < joint)
            # Is the simulation very inaccurate?
            sprime = int((logu - 1000.0) < joint)

            # Set the return values---minus=plus for all things here, since the
            # "tree" is of depth 0.
            thetaminus = thetaprime[:]
            thetaplus = thetaprime[:]
            rminus = rprime[:]
            rplus = rprime[:]
            gradminus = gradprime[:]
            gradplus = gradprime[:]

            # Compute the acceptance probability.
            alphaprime = min(1.0, np.exp(joint - joint0))
            nalphaprime = 1

            if v == 1 and traj is not None:
                ind_plus, ind_minus = ind + 1, ind
                traj.add_sample(thetaprime, ind_plus, which="plus")
                ind_prime = ind_plus
            elif traj is not None:
                ind_plus, ind_minus = ind, ind + 1
                traj.add_sample(thetaprime, ind_minus, which="minus")
                ind_prime = ind_minus
        else:
            # Recursion: Implicitly build the height j-1 left and right subtrees.
            if v == 1:
                (
                    thetaminus,
                    rminus,
                    gradminus,
                    thetaplus,
                    rplus,
                    gradplus,
                    thetaprime,
                    gradprime,
                    logpprime,
                    nprime,
                    sprime,
                    alphaprime,
                    nalphaprime,
                    ind_plus,
                    ind_minus,
                    ind_prime,
                ) = self.build_tree(theta, r, grad, logu, v, j - 1, epsilon, joint0, ind, traj, force_trajlen)
            else:
                (
                    thetaminus,
                    rminus,
                    gradminus,
                    thetaplus,
                    rplus,
                    gradplus,
                    thetaprime,
                    gradprime,
                    logpprime,
                    nprime,
                    sprime,
                    alphaprime,
                    nalphaprime,
                    ind_plus,
                    ind_minus,
                    ind_prime,
                ) = self.build_tree(theta, r, grad, logu, v, j - 1, epsilon, joint0, ind, traj, force_trajlen)

            # No need to keep going if the stopping criteria were met in the first subtree.
            if sprime == 1:
                if v == -1:
                    (
                        thetaminus,
                        rminus,
                        gradminus,
                        _,
                        _,
                        _,
                        thetaprime2,
                        gradprime2,
                        logpprime2,
                        nprime2,
                        sprime2,
                        alphaprime2,
                        nalphaprime2,
                        ind_plus,
                        ind_minus,
                        ind_prime2,
                    ) = self.build_tree(
                        thetaminus, rminus, gradminus, logu, v, j - 1, epsilon, joint0, ind_minus, traj, force_trajlen
                    )
                else:
                    (
                        _,
                        _,
                        _,
                        thetaplus,
                        rplus,
                        gradplus,
                        thetaprime2,
                        gradprime2,
                        logpprime2,
                        nprime2,
                        sprime2,
                        alphaprime2,
                        nalphaprime2,
                        ind_plus,
                        ind_minus,
                        ind_prime2,
                    ) = self.build_tree(
                        thetaplus, rplus, gradplus, logu, v, j - 1, epsilon, joint0, ind_plus, traj, force_trajlen
                    )
                # Choose which subtree to propagate a sample up from.
                if np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.0)):
                    thetaprime = thetaprime2[:]
                    gradprime = gradprime2[:]
                    logpprime = logpprime2
                    ind_prime = ind_prime2

                # Update the number of valid points.
                nprime = int(nprime) + int(nprime2)
                # Update the stopping criterion.
                sprime = int(
                    sprime
                    and sprime2
                    and self.stop_criterion(
                        thetaminus, thetaplus, rminus, rplus, force_trajlen, max(ind_plus, ind_minus)
                    )
                )
                # Update the acceptance probability statistics.
                alphaprime = alphaprime + alphaprime2
                nalphaprime = nalphaprime + nalphaprime2

        return (
            thetaminus,
            rminus,
            gradminus,
            thetaplus,
            rplus,
            gradplus,
            thetaprime,
            gradprime,
            logpprime,
            nprime,
            sprime,
            alphaprime,
            nalphaprime,
            ind_plus,
            ind_minus,
            ind_prime,
        )

    def __call__(self, x, iter, beta):
        """Take one HMC trajectory step"""
        super(NUTSJump, self).__call__(x, iter, beta)

        x = np.atleast_1d(x)
        if len(np.shape(x)) > 1:
            raise ValueError("x is expected to be a 1-D array")

        q = self.forward(x)

        self.beta = beta

        # Always start evaluating the distribution and gradient.
        # Potential speed-up: obtain these values from elsewhere, since we must
        # have evaluated them already?
        logp, grad = self.func_grad_white(q)

        if self.epsilon is None and self.force_epsilon is None:
            # First time doing an HMC jump
            self.epsilon = self.find_reasonable_epsilon(q, grad, logp)
            # print("Find reasonable epsilon: ", self.epsilon)
            self.mu = np.log(10.0 * self.epsilon)  # For dual averaging
        elif self.epsilon is None and self.force_epsilon is not None:
            # Force epsilon to be a certain value
            self.epsilon = self.force_epsilon
            self.mu = np.log(10.0 * self.epsilon)  # For dual averaging
        elif self.force_epsilon is not None:
            # Force epsilon to be a certain value
            self.epsilon = self.force_epsilon

        # Update the mass matrix when in burn-in stage
        # Currently, there is a problem with adjusting the mass matrix.
        # Stepsize and mass matrix need to be tuned together?
        # if iter <= self.nburn:
        #    self.update_cf()

        # Set the start of the trajectory
        r0 = self.draw_momenta()
        joint = self.loghamiltonian(logp, r0)

        # Initial slice sampling variable
        logu = float(joint - np.random.exponential(1, size=1))

        # Initialize the binary tree for this trajectory
        sample = np.copy(q)
        lnprob = np.copy(logp)
        thetaminus = np.copy(sample)
        thetaplus = np.copy(sample)
        rminus = np.copy(r0)
        rplus = np.copy(r0)
        gradminus = np.copy(grad)
        gradplus = np.copy(grad)

        j = 0  # Initial tree heigth j = 0
        n = 1  # Initially, the only valid point is the initial point
        s = 1  # Stopping criterion

        # Reset the trajectory buffer
        self.traj.reset()
        self.traj.add_sample(thetaminus, self.traj.length())
        trajind, trajind_minus, trajind_plus, trajind_prime = 0, 0, 0, 0

        while s == 1:
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)

            # Double the size of the tree.
            if v == -1:
                (
                    thetaminus,
                    rminus,
                    gradminus,
                    _,
                    _,
                    _,
                    thetaprime,
                    gradprime,
                    logpprime,
                    nprime,
                    sprime,
                    alpha,
                    nalpha,
                    trajind_plus,
                    trajind_minus,
                    trajind_prime,
                ) = self.build_tree(
                    thetaminus,
                    rminus,
                    gradminus,
                    logu,
                    v,
                    j,
                    self.epsilon,
                    joint,
                    trajind_minus,
                    self.traj,
                    self.force_trajlen,
                )
            else:
                (
                    _,
                    _,
                    _,
                    thetaplus,
                    rplus,
                    gradplus,
                    thetaprime,
                    gradprime,
                    logpprime,
                    nprime,
                    sprime,
                    alpha,
                    nalpha,
                    trajind_plus,
                    trajind_minus,
                    trajind_prime,
                ) = self.build_tree(
                    thetaplus,
                    rplus,
                    gradplus,
                    logu,
                    v,
                    j,
                    self.epsilon,
                    joint,
                    trajind_plus,
                    self.traj,
                    self.force_trajlen,
                )

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (np.random.uniform() < _tmp):
                sample[:] = thetaprime[:]
                lnprob = np.copy(logpprime)
                grad = np.copy(gradprime)
                trajind = trajind_prime

            # Update number of valid points we've seen.
            n += nprime
            # Decide if it's time to stop.
            s = sprime and self.stop_criterion(
                thetaminus, thetaplus, rminus, rplus, self.force_trajlen, max(trajind_plus, trajind_minus)
            )
            # Increment depth.
            j += 1

            sys.stdout.flush()

        # Do adaptation of epsilon if we're still doing burn-in.
        if self.force_epsilon is None:
            eta = 1.0 / float(self.iter + self.t0)
            self.Hbar = (1.0 - eta) * self.Hbar + eta * (self.delta - alpha / float(nalpha))

            if iter <= self.nburn:
                # Still in the burn-in phase. So adjust epsilon
                self.epsilon = np.exp(self.mu - np.sqrt(self.iter) / self.gamma * self.Hbar)
                eta = self.iter ** -self.kappa
                self.epsilonbar = np.exp((1.0 - eta) * np.log(self.epsilonbar) + eta * np.log(self.epsilon))

            else:
                self.epsilon = self.epsilonbar

        if self.trajectoryDir is not None:
            # Write the whole trajectory to file
            if iter <= self.nburn and self.write_burnin:
                trajfile_plus = os.path.join(self.trajectoryDir, "burnin-plus-{num:06d}.txt".format(num=iter))
                trajfile_minus = os.path.join(self.trajectoryDir, "burnin-minus-{num:06d}.txt".format(num=iter))
                trajfile_used = os.path.join(self.trajectoryDir, "burnin-used-{num:06d}.txt".format(num=iter))

                np.savetxt(trajfile_plus, self.traj.get_trajectory(which="plus")[0])
                np.savetxt(trajfile_minus, self.traj.get_trajectory(which="minus")[0])
                np.savetxt(trajfile_used, self.traj.get_used_trajectory(trajind))
            elif iter > self.nburn:
                trajfile_plus = os.path.join(self.trajectoryDir, "plus-{num:06d}.txt".format(num=iter - self.nburn))
                trajfile_minus = os.path.join(self.trajectoryDir, "minus-{num:06d}.txt".format(num=iter - self.nburn))
                trajfile_used = os.path.join(self.trajectoryDir, "used-{num:06d}.txt".format(num=iter - self.nburn))

                np.savetxt(trajfile_plus, self.traj.get_trajectory(which="plus")[0])
                np.savetxt(trajfile_minus, self.traj.get_trajectory(which="minus")[0])
                np.savetxt(trajfile_used, self.traj.get_used_trajectory(trajind))

        # We need to always accept this step, so the qxy is just the inverse MH ratio
        qxy = logp - lnprob

        return self.backward(sample), qxy
