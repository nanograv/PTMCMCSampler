# PTMCMCSampler
[![DOI](https://zenodo.org/badge/32821232.svg)](https://zenodo.org/badge/latestdoi/32821232)

MPI enabled Parallel Tempering MCMC code written in Python.

Please visit http://jellis18.github.io/PTMCMCSampler/ for documentation.


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

## Installation

The PTMCMCSampler is available from PyPI. The easiest way to install is via pip:
```
pip install PTMCMCSampler
```
### Python 3
In order for the sampler to run correctly with Python 3 kernels the GitHub version of acor needs to be installed. (Currently the PyPI version is behind the GitHub version.) It can be easily installed with:
```
pip install git+https://github.com/dfm/acor.git@master
```
