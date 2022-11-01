from importlib.metadata import PackageNotFoundError, version

from PTMCMCSampler import PTMCMCSampler  # noqa: F401

try:
    __version__ = version("ptmcmcsampler")
except PackageNotFoundError:
    # package is not installed
    pass
