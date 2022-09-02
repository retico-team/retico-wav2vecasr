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

exec(open("retico_wav2vecasr/version.py").read())

import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")


config = {
    "description": "The Huggingface wav2vec ASR incremental modules for the retico framework",
    "author": "Tahsin Mir Imtiaz, Ryan Whetten, Casey Kennington, Thilo Michael",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/retico-team/retico-wav2vecasr",
    "download_url": "https://github.com/retico-team/retico-wav2vecasr",
    "author_email": "caseykennington@boisestate.edu",
    "version": __version__,
    "install_requires": [
        "retico-core~=0.2",
        "transformers~=4.21",
        "webrtcvad~=2.0",
        "pydub~=0.25",
        "numpy~=1.23",
    ],
    "packages": find_packages(),
    "name": "retico-wav2vecasr",
    "keywords": "retico, framework, incremental, dialogue, dialog, asr, speech",
    "classifiers": [
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
}

setup(**config)
