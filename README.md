# PTMCMCSampler

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/nanograv/PTMCMCSampler)](https://github.com/nanograv/PTMCMCSampler/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/ptmcmcsampler)](https://pypi.org/project/ptmcmcsampler/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ptmcmcsampler.svg)](https://anaconda.org/conda-forge/ptmcmcsampler)
[![GitHub Workflow Status (event)](https://img.shields.io/github/workflow/status/nanograv/PTMCMCSampler/CI%20targets?label=CI%20Tests)](https://github.com/nanograv/PTMCMCSampler/actions/workflows/ci_test.yml)

[![DOI](https://zenodo.org/badge/32821232.svg)](https://zenodo.org/badge/latestdoi/32821232)
[![Python Versions](https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10%2C%203.11-blue.svg)]()
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/nanograv/PTMCMCSampler/blob/master/LICENSE)

**NOTE: This project was moved under the [NANOGrav][https://github.com/nanograv] github organization in November 2023**

MPI enabled Parallel Tempering MCMC code written in Python.

See the [examples](https://github.com/nanograv/PTMCMCSampler/tree/master/examples) for some simple use cases.

For MPI support you will need A functional MPI 1.x/2.x/3.x implementation like:

- [MPICH](http://www.mpich.org/)

  ```bash
  # mac
  brew install mpich

  # debian
  sudo apt install mpich
  ```

- [Open MPI](http://www.open-mpi.org/)

  ```bash
  # mac
  brew install open-mpi

  # debian
  sudo apt install libopenmpi-dev
  ```

To run with MPI support you can run your script containing a sampler with:

```bash
mpirun -np <number of temperature chains> script.py
```

This will kick off `np` chains running at different temperatures. The temperature ladder and sampling schemes can be set in the `PTMCMCSampler.sample()` method.

## Installation

### Development

For development clone this repo and run:

```bash
make init
source venv/bin/activate
```

### Via pip

```bash
pip install ptmcmcsampler
```

for MPI support use

```bash
pip install ptmcmcsampler[mpi]
```

### Via conda

```bash
conda install -c conda-forge ptmcmcsampler
```

for MPI support use

```bash
conda install -c conda-forge ptmcmcsampler mpi4py
```

#### Via Docker

**Production**:

For production use, the latest release of PTMCMCSampler is installed directly from PyPi.
```
# Build the image, tagging it as `ptmcmc:latest`.  
# 
# This example includes both optional MPI support and the optional `acor` library 
# (you can omit either / both).
docker build --pull --build-arg "MPI=1" --build-arg "ACOR=1" -t ptmcmc --no-cache .

# Run the image, mounting the `data/` subdirectory on your computer 
# to `/code/data/` inside the Docker container. Note that MPI won't work 
# if we run as the root user (the default).
docker run -it --rm --user mcmc_user -v $(pwd)/data:/code/data ptmcmc

# When finished with the container, exit back to the host OS
CTRL^D
```


**Development**: 

For PTMCMCSampler development, dependencies are installed from `requirements.txt` and 
`requirements_dev.txt`. 
No PTMCMCSampler code is omitted from the built image, whose purpose is for testing new code. 
You can also add `--build-arg "ACOR=1"` to the build command to include the optional `acor` 
dependency (MPI is always included via `requirements.txt`).

```bash
# Build the image
docker build --pull -t ptmcmc --build-arg "TARGET_ENV=dev" --no-cache .

# Run the image, mounting the working directory on the host OS to /code/ inside the container.
# MPI won't work if we run as the root user (the default).
docker run -it --rm --user mcmc_user -v $(pwd)/:/code ptmcmc

# Exit back to host OS
CTRL-D
```






## Attribution

If you make use of this code, please cite:

```
@misc{justin_ellis_2017_1037579,
  author       = {Justin Ellis and
                  Rutger van Haasteren},
  title        = {jellis18/PTMCMCSampler: Official Release},
  month        = oct,
  year         = 2017,
  doi          = {10.5281/zenodo.1037579},
  url          = {https://doi.org/10.5281/zenodo.1037579}
}
```

### Correlation Length

In order for the sampler to run correctly using `acor` with Python 3 kernels the GitHub version of acor needs to be installed. (Currently the PyPI version is behind the GitHub version.) It can be easily installed with:

```
pip install git+https://github.com/dfm/acor.git@master
```

> Note that `acor` is not required to run the sampler, it simply calculates the effective chain length for output in the chain file.
