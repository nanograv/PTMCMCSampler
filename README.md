# PTMCMCSampler
[![DOI](https://zenodo.org/badge/32821232.svg)](https://zenodo.org/badge/latestdoi/32821232)
[![Python Versions](https://img.shields.io/badge/python-3.6%2C%203.7%2C%203.8%2C%203.9-blue.svg)]()
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/jellis18/PTMCMCSampler/blob/master/LICENSE)



MPI enabled Parallel Tempering MCMC code written in Python.

See the [examples](https://github.com/jellis18/PTMCMCSampler/tree/master/examples) for some simple use cases.

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
