###################################################################################################
# Builds a Docker image for running or developing PTMCMCSampler.
###################################################################################################
# Note that Python dependency versions are not pinned in requirements*.txt files,
# so rebuilds of the image over time may generate different results.
#
# ACOR: provide any value (e.g. --build-arg "ACOR=1") to install optional acor library and
#       its unique binary dependencies
ARG ACOR
# MPI: provide any value (e.g. --build-arg "MPI=1") to install optional mpi4py library and
#       its unique binary dependencies.  This argument is ignored when TARGET_ENV=dev, since
#       requirements.txt specifies mpi4py as a requirement (so it's always installed in that case)
ARG MPI
# Target environment:
#    * "prod" = install released PTMCMCSampler code from PyPi for production use.
#    * "dev" = do not install PTMCMCSampler, only its dependencies to support development work.
ARG TARGET_ENV="prod"
ARG VIRTUAL_ENV=/home/mcmc_user/.venv


###################################################################################################
# Base build stage installs common dependencies
###################################################################################################
FROM library/python:3.11-slim-bullseye as build-base
ARG MPI
ARG TARGET_ENV
ARG VIRTUAL_ENV

# set environment used here and inherited by derived build stages
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    # Set environment variables to simulate activation of a Python venv
    # https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
    VIRTUAL_ENV=$VIRTUAL_ENV \
    PATH=${VIRTUAL_ENV}/bin:${PATH} \
    # Avoid cryptic openmpi read errors that would otherwise result from running < 5.0 in Docker.
    # https://github.com/open-mpi/ompi/issues/4948#issuecomment-1221061698.
    # See also https://github.com/jellis18/PTMCMCSampler/issues/23.
    # At the time of writing (8/30/22), the version installed by apt-get in bullseye is 4.1.0-10.
    # This may theoretically hurt performance, but doesn't appear to.
    # It is safer than the alternative of enabling Docker's SYS_PTRACE capability.
    OMPI_MCA_btl_vader_single_copy_mechanism="none"

WORKDIR /tmp

RUN set -ex \
 # add a user and group so we don't have problems executing mpirun as root
 && addgroup --gid 1000 --system mcmc_user \
 && useradd --uid 1000 --gid 1000 --create-home mcmc_user \
 && if [ -n "$MPI" ] || [ "$TARGET_ENV" = "dev" ]; then \
    apt-get update \
      && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends openmpi-bin; \
 fi \
 # clean up cache after running apt-get update \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir /code/ \
 && chown -R mcmc_user:mcmc_user /code/

###################################################################################################
# Intermediate stage adds install-time system binary dependencies
###################################################################################################
FROM build-base as install-base
ARG ACOR
ARG MPI
ARG TARGET_ENV

# Add install-time dependency gcc to support mpi4py install
RUN apt-get update \
 && if [ -n "$MPI" ] || [ "$TARGET_ENV" = "dev" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
        gcc \
        libopenmpi-dev; \
 fi \
 # If installing optional acor package, add additional binaries that support installing it
 && if [ -n "$ACOR" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    # Install Git so we can clone newer acor from GitHub (as suggested in README.md)
    git \
    python3-dev \
    g++; \
 fi \
 # clean up cache after running apt-get update \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && rm -rf /var/lib/apt/lists/*

###################################################################################################
# Development stage installs PTMCMCSampler dependencies
# from requirements.txt and requirements_dev.txt
###################################################################################################
FROM install-base as dev-preinstall
ARG VIRTUAL_ENV

# for PTMCMCSampler development, install only the dependencies
COPY requirements.txt requirements_dev.txt /tmp/

# create venv for installing Python packages
RUN python -m venv $VIRTUAL_ENV \
    && pip install --upgrade pip \
    # install PTMCMCSampler's Python dependencies
    && pip install -r requirements.txt \
    && pip install -r requirements_dev.txt \
    # purge the pip cache to keep image size small
    && pip cache purge \
    && rm -rf root/.cache/ \
    # Remove Python bytecode (causes image bloat)
    && find $VIRTUAL_ENV -name __pycache__ | xargs rm -rf

###################################################################################################
# Production stage installs PTMCMCSampler & dependencies from pypi
###################################################################################################
FROM install-base as prod-preinstall
ARG ACOR
ARG MPI
ARG VIRTUAL_ENV

# To run a released version of the code, install PTMCMCSampler and dependencies from pypi
# create venv for installing Python packages
RUN python -m venv $VIRTUAL_ENV \
  && pip install --upgrade pip \
  && if [ -n "$MPI" ]; then \
    pip install ptmcmcsampler[mpi]; \
  else \
    pip install ptmcmcsampler; \
  fi \
  # Install optional acor package from GitHub
  && if [ -n "$ACOR" ]; then \
    pip install git+https://github.com/dfm/acor.git@master; \
  fi \
  # purge the pip cache to keep image size small
  && pip cache purge \
  && rm -rf root/.cache/ \
  # Remove Python bytecode (causes image bloat)
  && find $VIRTUAL_ENV -name __pycache__ | xargs rm -rf

###################################################################################################
# Intermediate stage selects which prior stage to use later.
# Stand-in for lack of variable expansion support in COPY --from
###################################################################################################
FROM ${TARGET_ENV}-preinstall as conditional-install

###################################################################################################
# Final stage pulls in target environment-specific dependencies
###################################################################################################
FROM build-base as final
ARG VIRTUAL_ENV

# Copy in the Python virtualenv from the targeted intermediate build stage,
# omitting all install-time dependencies and any other artifacts
# from intermediate build stages.
COPY --from=conditional-install $VIRTUAL_ENV $VIRTUAL_ENV

WORKDIR /code/
ENTRYPOINT ["/bin/bash"]
