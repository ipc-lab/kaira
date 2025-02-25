Installation
============

Overview
--------
This guide provides comprehensive instructions for installing Kaira on your system. Kaira is designed to work across multiple platforms with minimal setup required.

Prerequisites
------------
- Python 3.8 or higher
- pip (Python package installer)
- 2GB+ of free disk space
- (Optional) CUDA-compatible GPU for accelerated processing

Installation Methods
-------------------

Quick Installation
~~~~~~~~~~~~~~~~~
The fastest way to install Kaira is directly from PyPI:

.. code-block:: bash

   pip install kaira

From Source
~~~~~~~~~~
For the latest features or contributions, install from source:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/ipc-lab/kaira.git
      cd kaira

2. Install the package:

   .. code-block:: bash

      pip install .

Using Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
--------------------

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
~~~~~~~~~~~~~~~
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
---------------------
Confirm your installation is working correctly:

.. code-block:: bash

   python -c "import kaira; print(f'Kaira version {kaira.__version__} successfully installed')"

This should display your installed version without any errors.

Troubleshooting
--------------
Common Issues:

- **"ImportError: No module named kaira"**: Make sure your virtual environment is activated or reinstall using ``pip install --force-reinstall kaira``
- **Permission errors**: Use ``pip install --user kaira`` or create a virtual environment
- **Dependency conflicts**: Try installing in a fresh virtual environment
- **GPU not detected**: Verify your CUDA installation with ``python -c "import torch; print(torch.cuda.is_available())"``

For more help, see our :doc:`faq` or join our `community forum <https://github.com/ipc-lab/kaira/discussions>`_.

Uninstallation
-------------
If you need to remove Kaira:

.. code-block:: bash

   pip uninstall kaira
