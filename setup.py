"""
Setup script.

Use this script to install the Huggingface wav2vec ASR incremental modules for the retico framework.
Usage:
    $ python3 setup.py install
The run the simulation:
    $ retico [-h]
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

config = {
    "description": "The Huggingface wav2vec ASR incremental modules for the retico framework",
    "author": "Tahsin Mir Imtiaz, Ryan Whetten, Casey Kennington",
    "url": "??",
    "download_url": "??",
    "author_email": "caseykennington@boisestate.edu",
    "version": "0.1",
    "install_requires": ["retico-core~=0.2.0", "torch~=1.11.0", "hugginface~=0.6.0", "transformers", "webrtcvad", "colorama"],
    "packages": find_packages(),
    "name": "retico-wav2vecasr",
}

setup(**config)