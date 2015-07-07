import os
import sys
import numpy

from setuptools import setup
from setuptools import Extension

import PTMCMCSampler


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


setup(
    name="PTMCMCSampler",
    version=PTMCMCSampler.__version__,
    author="Justin A. Ellis",
    author_email="justin.ellis18@gmail.com",
    packages=["PTMCMCSampler"],
    package_dir={'PTMCMCSampler': 'PTMCMCSampler'},
    url="https://github.com/jellis18/PTMCMCSampler",
    license="GPLv3",
    zip_safe=False,
    description="Parallel tempering MCMC sampler written in Python",
    long_description=open("README.md").read() + "\n\n"
                    + "---------\n\n"
                    + open("HISTORY.md").read(),
    package_data={"": ["README.md", "HISTORY.md"]},
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
)
