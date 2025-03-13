.. image:: docs/_static/logo.png
    :align: center
    :alt: Kaira Framework Logo

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

.. image:: https://badges.gitter.im//community.svg
   :target: https://gitter.im/kaira/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link
   :alt: Gitter Chat

.. image:: https://dev.azure.com/ipc-lab/kaira/_apis/build/status/ipc-lab.kaira?branchName=master
   :target: https://dev.azure.com/ipc-lab/kaira/_build/latest?branchName=master
   :alt: Azure Pipelines Build Status

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

Overview
--------
This guide provides comprehensive instructions for installing Kaira on your system. Kaira is designed to work across multiple platforms with minimal setup required.

Prerequisites
-------------
- Python 3.8 or higher
- pip (Python package installer)
- (Optional) CUDA-compatible GPU for accelerated processing

Installation Methods
--------------------

Quick Installation
~~~~~~~~~~~~~~~~~~
The fastest way to install Kaira is directly from PyPI:

.. code-block:: bash

   pip install kaira

From Source
~~~~~~~~~~~
For the latest features or contributions, install from source:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/ipc-lab/kaira.git
      cd kaira

2. Install the package:

   .. code-block:: bash

      pip install .

Using Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a cleaner installation that won't interfere with other Python packages:

.. code-block:: bash

   # Create a virtual environment
   python -m venv kaira-env

   # Activate the environment
   # On Linux/macOS:
   source kaira-env/bin/activate

   # On Windows:
   kaira-env\Scripts\activate

   # Install Kaira
   pip install kaira

System-Specific Notes
---------------------

Windows
~~~~~~~
- You may need to run the Command Prompt or PowerShell as administrator
- If you encounter path issues, ensure Python is added to your PATH environment variable

macOS
~~~~~
- You may need to use ``python3`` explicitly instead of ``python``
- Some users may need to install XCode command line tools first: ``xcode-select --install``

Linux
~~~~~
- Ensure you have the required build tools: ``sudo apt-get install build-essential python3-dev`` (Ubuntu/Debian)

GPU Acceleration
~~~~~~~~~~~~~~~~
Kaira automatically detects and utilizes available CUDA-compatible GPUs. For GPU support, you need:

1. Installed the appropriate NVIDIA drivers for your GPU
2. Installed a compatible version of CUDA Toolkit
3. Properly set up your system environment variables
4. PyTorch with GPU support

To install Kaira with GPU support, we recommend following the PyTorch installation instructions first to ensure proper CUDA compatibility:

.. code-block:: bash

   # Check PyTorch website for the specific command for your system and CUDA version
   # https://pytorch.org/get-started/locally/
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Example for CUDA 11.8

   # Then install Kaira
   pip install kaira

For more details on PyTorch GPU configuration, please refer to the `PyTorch documentation <https://pytorch.org/docs/stable/notes/cuda.html>`_.

No additional Kaira-specific installation commands are required for GPU support as it's included in the main package.

Verifying Installation
----------------------
Confirm your installation is working correctly:

.. code-block:: bash

   python -c "import kaira; print(f'Kaira version {kaira.__version__} successfully installed')"

This should display your installed version without any errors.

Troubleshooting
---------------
Common Issues:

- **"ImportError: No module named kaira"**: Make sure your virtual environment is activated or reinstall using ``pip install --force-reinstall kaira``
- **Permission errors**: Use ``pip install --user kaira`` or create a virtual environment
- **Dependency conflicts**: Try installing in a fresh virtual environment
- **GPU not detected**: Verify your CUDA installation with ``python -c "import torch; print(torch.cuda.is_available())"``

For more help, see our :doc:`faq` or join our `community forum <https://github.com/ipc-lab/kaira/discussions>`_.

Uninstallation
--------------
If you need to remove Kaira:

.. code-block:: bash

   pip uninstall kaira

Quick Links
===========

- **GitHub Repository:** `https://github.com/ipc-lab/kaira/ <https://github.com/ipc-lab/kaira/>`_
- **PyPI Package:** `https://pypi.org/project/kaira <https://pypi.org/project/kaira/>`_
- **Travis CI:** `https://travis-ci.com/github/ipc-lab/kaira <https://travis-ci.com/github/ipc-lab/kaira>`_
- **Azure Pipelines:** `https://dev.azure.com/ipc-lab/kaira/ <https://dev.azure.com/ipc-lab/kaira/>`_
- **Circle CI:** `https://circleci.com/gh/ipc-lab/kaira/ <https://circleci.com/gh/ipc-lab/kaira/>`_
- **Appveyor:** `https://ci.appveyor.com/project/ipc-lab/kaira/branch/master <https://ci.appveyor.com/project/ipc-lab/kaira/branch/master>`_
- **Coveralls:** `https://coveralls.io/github/ipc-lab/kaira?branch=master <https://coveralls.io/github/ipc-lab/kaira?branch=master>`_
- **License:** `https://github.com/ipc-lab/kaira/blob/master/LICENSE <https://github.com/ipc-lab/kaira/blob/master/LICENSE>`_

Support
-------
Get help and connect with the Kaira community through these channels:

* `Documentation <https://kaira.readthedocs.io/>`_ - Official project documentation
* `GitHub Issues <https://github.com/ipc-lab/kaira/issues>`_ - Bug reports and feature requests
* `Discussions <https://github.com/ipc-lab/kaira/discussions>`_ - General questions and community discussions
* `Gitter Chat <https://gitter.im/ipc-lab/kaira>`_ - Live chat with developers and users

License
=======

Kaira is distributed under the terms of the `MIT License <https://github.com/ipc-lab/kaira/blob/master/LICENSE>`_.

Citing Kaira
============

For academic publications or any research work that makes use of Kaira, please acknowledge the repository by citing it using the BibTeX entry shown below:

.. code-block:: bibtex

    @misc{kaira,
        author       = {Selim F. Yilmaz and Imperial IPC Lab},
        title        = {Kaira},
        year         = {2025},
        howpublished = {\url{https://github.com/ipc-lab/kaira}},
        note         = {Accessed: 1 March 2025 (TODO: UPDATE)},
    }
