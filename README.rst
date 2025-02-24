.. image:: docs/_static/logo.png
    :align: center

Kaira — Wireless Communication Simulation Toolbox for PyTorch
==============================================================

.. image:: https://img.shields.io/pypi/v/kaira
   :target: https://pypi.org/project/kaira/
   :alt: PyPI Version

.. image:: https://img.shields.io/github/v/release/ipc-lab/kaira
   :target: https://github.com/ipc-lab/kaira/releases
   :alt: GitHub Release (Latest)

.. image:: https://readthedocs.org/projects/kaira/badge/?version=latest
   :target: https://kaira.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://badges.gitter.im/ipc-lab/kaira/community.svg
   :target: https://gitter.im/ipc-lab/kaira/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link
   :alt: Gitter Chat

.. image:: https://dev.azure.com/ipc-lab/kaira/_apis/build/status/ipc-lab.kaira?branchName=master
   :target: https://dev.azure.com/ipc-lab/kaira/_build/latest?branchName=master
   :alt: Azure Pipelines Build Status

.. image:: https://travis-ci.org/ipc-lab/kaira.svg?branch=master
   :target: https://travis-ci.org/ipc-lab/kaira
   :alt: Travis CI Build Status

.. image:: https://ci.appveyor.com/api/projects/status/<APPVEYOR_ID>/branch/master?svg=true
   :target: https://ci.appveyor.com/project/ipc-lab/kaira/branch/master
   :alt: AppVeyor Build Status

.. image:: https://circleci.com/gh/ipc-lab/kaira.svg?style=svg
   :target: https://circleci.com/gh/ipc-lab/kaira
   :alt: CircleCI Status

.. image:: https://coveralls.io/repos/github/ipc-lab/kaira/badge.svg?branch=master
   :target: https://coveralls.io/github/ipc-lab/kaira?branch=master
   :alt: Coverage Status

.. image:: https://img.shields.io/pypi/pyversions/kaira
   :target: https://github.com/ipc-lab/kaira/
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/badge/platforms-linux--64%2Cosx--64%2Cwin--64-green
   :target: https://github.com/ipc-lab/kaira/
   :alt: Supported Platforms

.. image:: https://img.shields.io/github/license/ipc-lab/kaira.svg
   :target: https://github.com/ipc-lab/kaira/blob/master/LICENSE
   :alt: License

Kaira is an open-source simulation toolkit for wireless communications built on PyTorch. It provides a modular, user-friendly platform for developing, testing, and benchmarking advanced wireless transmission algorithms—including deep learning-based approaches such as deep joint source-channel coding (DeepJSCC). Designed to accelerate research and innovation, Kaira integrates seamlessly with existing PyTorch projects, supporting rapid prototyping of novel communication strategies.

`Documentation <https://kaira.readthedocs.io/en/latest/>`__
DeepJSCC
========

Deep joint source-channel coding (DeepJSCC) is a wireless data transmission method that uses deep neural networks to directly map data to channel input symbols, bypassing the need for explicit compression or error correction codes. This end-to-end framework employs two convolutional neural networks as encoder and decoder, functioning like an autoencoder with a non-trainable layer representing the noisy communication channel. Deep JSCC surpasses traditional digital transmission methods in low signal-to-noise ratio and bandwidth conditions, and gracefully degrades performance as the channel signal-to-noise ratio changes. It can also learn to resist noise and outperform traditional digital communication in slow Rayleigh fading channels.

Features
========

1. **Research-Oriented**: Designed to accelerate deep joint-source channel coding research.
2. **Versatility**: Compatible with various data types and neural network architectures.
3. **Ease of Use**: User-friendly and easy to integrate with existing PyTorch projects.
4. **Open Source**: Allows for community contributions and improvements.
5. **Well Documented**: Comes with comprehensive documentation for easy understanding.

Installation
============

Overview:
---------
Follow these steps to install and set up Kaira for development and experimentation.

Prerequisites:
- Python 3.8 or higher
- Sphinx (for building the documentation)
- Virtual environment (recommended)

Installation Steps:
-------------------
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

4. (Optional) Install additional development tools:

   .. code-block:: bash

      pip install -r dev-requirements.txt

5. Build the documentation:

   .. code-block:: bash

      make html

Additional Notes:
-----------------
For advanced usage, see the API Reference section. If you encounter issues, review the README.md or consult the community for support.

Quick Links
===========

- **GitHub Repository:** `https://github.com/ipc-lab/kaira/ <https://github.com/ipc-lab/kaira/>`_
- **Documentation:** `https://kaira.readthedocs.io/ <https://kaira.readthedocs.io/>`_
- **PyPI Package:** `https://pypi.org/project/kaira <https://pypi.org/project/kaira/>`_
- **Travis CI:** `https://travis-ci.com/github/ipc-lab/kaira <https://travis-ci.com/github/ipc-lab/kaira>`_
- **Azure Pipelines:** `https://dev.azure.com/ipc-lab/kaira/ <https://dev.azure.com/ipc-lab/kaira/>`_
- **Circle CI:** `https://circleci.com/gh/ipc-lab/kaira/ <https://circleci.com/gh/ipc-lab/kaira/>`_
- **Appveyor:** `https://ci.appveyor.com/project/ipc-lab/kaira/branch/master <https://ci.appveyor.com/project/ipc-lab/kaira/branch/master>`_
- **Coveralls:** `https://coveralls.io/github/ipc-lab/kaira?branch=master <https://coveralls.io/github/ipc-lab/kaira?branch=master>`_
- **License:** `https://github.com/ipc-lab/kaira/blob/master/LICENSE <https://github.com/ipc-lab/kaira/blob/master/LICENSE>`_

Versioning
==========

Kaira adheres to [Semantic Versioning](http://semver.org/). Version numbers follow the pattern `major.minor.patch`, where:

- **major** versions introduce incompatible API changes,
- **minor** versions add functionality in a backwards-compatible manner,
- **patch** versions include backwards-compatible bug fixes.

This systematic approach ensures clear communication of changes and helps maintain compatibility.

License
=======

Kaira is distributed under the terms of the `MIT License <https://github.com/ipc-lab/kaira/blob/master/LICENSE>`_.

The full text of the license is shown below:


   :language: text
   :linenos:


Citing Kaira
============

For academic publications or any research work that makes use of Kaira, please acknowledge the repository by citing it using the BibTeX entry shown below:

.. code-block:: bibtex

    @misc{kaira,
        author       = {Selim F. Yilmaz and Imperial IPC Lab},
        title        = {Kaira},
        year         = {2025},
        howpublished = {\url{https://github.com/ipc-lab/kaira}},
        note         = {Accessed: 23 February 2025 (TODO: UPDATE)},
    }
