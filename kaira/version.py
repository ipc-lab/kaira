"""
Kaira: A Toolbox for Simulating Wireless Communication Systems
================================================================

Kaira focuses on wireless data transmission using deep joint
source-channel coding (DeepJSCC), providing simulation tools for
advanced wireless transmission algorithms that integrate with modern
deep learning frameworks.

License: MIT

Versioning
----------
This module uses a PEP 440 compliant versioning scheme.

Generic release markers:
    - Major.Minor       (e.g., "1.2")
    - Major.Minor.Patch (e.g., "1.2.3") for bugfix releases

Pre-release markers:
    - Alpha:    Major.Minor aN (e.g., "1.2a1")
    - Beta:     Major.Minor bN (e.g., "1.2b1")
    - RC:       Major.Minor rcN (e.g., "1.2rc1")

Development markers:
    - Dev:      Major.Minor.dev or Major.Minor.devN (where N is an integer; canonical form is "X.Y.dev0")

For example:
    "0.1.0", "2.0rc1", "2.1.dev0", etc.
"""

__version_info__ = (0, 1, 0)
__version__ = ".".join(map(str, __version_info__))

if __name__ == "__main__":
    print("Kaira version:", __version__)
