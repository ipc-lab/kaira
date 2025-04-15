![Kaira Framework Logo](docs/_static/logo.png){.align-center}

# Kaira --- Wireless Communication Simulation Toolbox for PyTorch

[![Python CI](https://github.com/ipc-lab/kaira/actions/workflows/ci.yml/badge.svg)](https://github.com/ipc-lab/kaira/actions/workflows/ci.yml)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

[![Documentation Build Status](https://github.com/ipc-lab/kaira/actions/workflows/docs.yml/badge.svg)](https://github.com/ipc-lab/kaira/actions/workflows/docs.yml)

[![ReadTheDocs Status](https://readthedocs.org/projects/kaira/badge/?version=latest)](https://kaira.readthedocs.io/en/latest/?badge=latest)

[![PyPI Version](https://img.shields.io/pypi/v/kaira)](https://pypi.org/project/kaira/)

[![GitHub Release (Latest)](https://img.shields.io/github/v/release/ipc-lab/kaira)](https://github.com/ipc-lab/kaira/releases)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kaira)](https://github.com/ipc-lab/kaira/)

[![Supported Platforms](https://img.shields.io/badge/platforms-linux--64%2Cosx--64%2Cwin--64-green)](https://github.com/ipc-lab/kaira/)

[![License](https://img.shields.io/github/license/ipc-lab/kaira.svg)](https://github.com/ipc-lab/kaira/blob/master/LICENSE)

[![Coverage Status](https://coveralls.io/repos/github/ipc-lab/kaira/badge.svg?branch=master)](https://coveralls.io/github/ipc-lab/kaira?branch=master)

[![Azure Pipelines Build Status](https://dev.azure.com/ipc-lab/kaira/_apis/build/status/ipc-lab.kaira?branchName=master)](https://dev.azure.com/ipc-lab/kaira/_build/latest?branchName=master)

[![Dependabot Updates](https://github.com/ipc-lab/kaira/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/ipc-lab/kaira/actions/workflows/dependabot/dependabot-updates)

[![Gitter Chat](https://badges.gitter.im//community.svg)](https://gitter.im/kaira/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

Kaira is an open-source simulation toolkit for communications research
built on PyTorch. It provides a modular, user-friendly platform for
developing, testing, and benchmarking advanced wireless transmission
algorithms---including deep learning-based approaches such as deep joint
source-channel coding (DeepJSCC). Designed to accelerate research and
innovation, Kaira integrates seamlessly with existing PyTorch projects,
supporting rapid prototyping of novel communication strategies.

[Documentation](https://kaira.readthedocs.io/en/latest/) Features
========

1.  **Research-Oriented**: Designed to accelerate communications
    research.
2.  **Versatility**: Compatible with various data types and neural
    network architectures.
3.  **Ease of Use**: User-friendly and easy to integrate with existing
    PyTorch projects.
4.  **Open Source**: Allows for community contributions and
    improvements.
5.  **Well Documented**: Comes with comprehensive documentation for easy
    understanding.

# Installation

## Overview

This guide provides comprehensive instructions for installing Kaira on
your system. Kaira is designed to work across multiple platforms with
minimal setup required.

## Prerequisites

-   Python 3.8 or higher
-   pip (Python package installer)
-   (Optional) CUDA-compatible GPU for accelerated processing

## Installation Methods

### Quick Installation

The fastest way to install Kaira is directly from PyPI:

``` bash
pip install kaira
```

### From Source

For the latest features or contributions, install from source:

1.  Clone the repository:

    ``` bash
    git clone https://github.com/ipc-lab/kaira.git
    cd kaira
    ```

2.  Install the package:

    ``` bash
    pip install .
    ```

### Using Virtual Environment (Recommended)

For a cleaner installation that won\'t interfere with other Python
packages:

``` bash
# Create a virtual environment
python -m venv kaira-env

# Activate the environment
# On Linux/macOS:
source kaira-env/bin/activate

# On Windows:
kaira-env\Scripts\activate

# Install Kaira
pip install kaira
```

## System-Specific Notes

### Windows

-   You may need to run the Command Prompt or PowerShell as
    administrator
-   If you encounter path issues, ensure Python is added to your PATH
    environment variable

### macOS

-   You may need to use `python3` explicitly instead of `python`
-   Some users may need to install XCode command line tools first:
    `xcode-select --install`

### Linux

-   Ensure you have the required build tools:
    `sudo apt-get install build-essential python3-dev` (Ubuntu/Debian)

### GPU Acceleration

Kaira automatically detects and utilizes available CUDA-compatible GPUs.
For GPU support, you need:

1.  Installed the appropriate NVIDIA drivers for your GPU
2.  Installed a compatible version of CUDA Toolkit
3.  Properly set up your system environment variables
4.  PyTorch with GPU support

To install Kaira with GPU support, we recommend following the PyTorch
installation instructions first to ensure proper CUDA compatibility:

``` bash
# Check PyTorch website for the specific command for your system and CUDA version
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Example for CUDA 11.8

# Then install Kaira
pip install kaira
```

For more details on PyTorch GPU configuration, please refer to the
[PyTorch
documentation](https://pytorch.org/docs/stable/notes/cuda.html).

No additional Kaira-specific installation commands are required for GPU
support as it\'s included in the main package.

## Verifying Installation

Confirm your installation is working correctly:

``` bash
python -c "import kaira; print(f'Kaira version {kaira.__version__} successfully installed')"
```

This should display your installed version without any errors.

## Troubleshooting

Common Issues:

-   **\"ImportError: No module named kaira\"**: Make sure your virtual
    environment is activated or reinstall using
    `pip install --force-reinstall kaira`
-   **Permission errors**: Use `pip install --user kaira` or create a
    virtual environment
-   **Dependency conflicts**: Try installing in a fresh virtual
    environment
-   **GPU not detected**: Verify your CUDA installation with
    `python -c "import torch; print(torch.cuda.is_available())"`

For more help, see our `faq`{.interpreted-text role="doc"} or join our
[community forum](https://github.com/ipc-lab/kaira/discussions).

## Uninstallation

If you need to remove Kaira:

``` bash
pip uninstall kaira
```

# Quick Links

-   **GitHub Repository:** <https://github.com/ipc-lab/kaira/>
-   **PyPI Package:**
    [https://pypi.org/project/kaira](https://pypi.org/project/kaira/)
-   **Travis CI:** <https://travis-ci.com/github/ipc-lab/kaira>
-   **Azure Pipelines:** <https://dev.azure.com/ipc-lab/kaira/>
-   **Circle CI:** <https://circleci.com/gh/ipc-lab/kaira/>
-   **Appveyor:**
    <https://ci.appveyor.com/project/ipc-lab/kaira/branch/master>
-   **Coveralls:**
    <https://coveralls.io/github/ipc-lab/kaira?branch=master>
-   **License:** <https://github.com/ipc-lab/kaira/blob/master/LICENSE>

## Support

Get help and connect with the Kaira community through these channels:

-   [Documentation](https://kaira.readthedocs.io/) - Official project
    documentation
-   [GitHub Issues](https://github.com/ipc-lab/kaira/issues) - Bug
    reports and feature requests
-   [Discussions](https://github.com/ipc-lab/kaira/discussions) -
    General questions and community discussions
-   [Gitter Chat](https://gitter.im/ipc-lab/kaira) - Live chat with
    developers and users

# License

Kaira is distributed under the terms of the [MIT
License](https://github.com/ipc-lab/kaira/blob/master/LICENSE).

# Citing Kaira

For academic publications or any research work that makes use of Kaira,
please acknowledge the repository by citing it using the BibTeX entry
shown below:

``` bibtex
@misc{kaira,
    author       = {Selim F. Yilmaz and Imperial IPC Lab},
    title        = {Kaira},
    year         = {2025},
    howpublished = {\url{https://github.com/ipc-lab/kaira}},
    note         = {Accessed: 1 March 2025 (TODO: UPDATE)},
}
```
