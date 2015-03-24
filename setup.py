import os
import sys
import numpy

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


setup(
    name="PTMCMCSampler",
    version='2015.01',
    author="Justin A. Ellis",
    author_email="justin.ellis18@gmail.com",
    packages=["PTMCMCSampler"],
    url="https://github.com/jellis18/PTMCMCSampler",
    license="GPLv3",
    description="Parallel tempering MCMC sampler written in Python",
    long_description=open("README.md").read() + "\n\n"
                    + "---------\n\n"
                    + open("HISTORY.md").read(),
    package_data={"": ["README.md", "HISTORY.md"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "mpi4py"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
)
