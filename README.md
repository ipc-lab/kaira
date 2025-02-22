<p align="center">
  <img src="docs/logo.png" />
</p>

# Kaira -- Wireless simulation toolbox for PyTorch

Kaira is a toolbox for simulating wireless communication systems, focused on wireless data transmission using deep joint source-channel coding (DeepJSCC). It is built on top of PyTorch and provides a simple and flexible framework for deep learning-based image transmission. Kaira is designed to accelerate research in wireless communication systems and facilitate the development of new algorithms. It is also easy to use and can be integrated with existing PyTorch projects.

## Table of Contents
- [DeepJSCC](#deepjscc)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Running Documentation](#running-documentation)
- [Contributing](#contributing)
- [Versioning](#versioning)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## DeepJSCC
Deep joint source-channel coding (DeepJSCC) is a wireless data transmission method that uses deep neural networks to directly map data to channel input symbols, bypassing the need for explicit compression or error correction codes. This end-to-end framework employs two convolutional neural networks as encoder and decoder, functioning like an autoencoder with a non-trainable layer representing the noisy communication channel. Deep JSCC surpasses traditional digital transmission methods in low signal-to-noise ratio and bandwidth conditions, and gracefully degrades performance as the channel signal-to-noise ratio changes. It can also learn to resist noise and outperform traditional digital communication in slow Rayleigh fading channels.

## Features

1. **Research-Oriented**: Designed to accelerate deep joint-source channel coding research.
2. **Versatility**: Compatible with various data types and neural network architectures.
3. **Ease of Use**: User-friendly and easy to integrate with existing PyTorch projects.
4. **Open Source**: Allows for community contributions and improvements.
5. **Well Documented**: Comes with comprehensive documentation for easy understanding.

## Installation

To install the latest version of Kaira, please run the following command:

```bash
pip install kaira
```

## Usage

To use kaira in your project, import it as follows:

```python
import kaira
```

## Examples

We have provided several examples in the `examples/` directory to help you get started.

## Running Documentation

To build and view the project documentation using Sphinx, follow these steps:

1. **Open a Terminal**  
   In Visual Studio Code, open the integrated terminal.

2. **Change Directory to Docs**  
   Navigate to the documentation folder by running:
   ```bash
   cd docs
   ```

3. **Build the Documentation**  
   - On Linux/macOS, run:
     ```bash
     make html
     ```
   - On Windows, run:
     ```bat
     make.bat html
     ```

4. **View the Output**  
   After the build completes, open the generated HTML files located in the `_build/html` directory. For example, view the main page by opening:
   ```plaintext
   index.html
   ```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to contribute.

## Versioning
We use [Semantic versioning](http://semver.org/) for this project.

## License

This project is licensed under the MIT License. See `LICENSE.md` for more details.

## Contact

If you have any questions or feedback, please feel free to open an issue or submit a pull request.

## Acknowledgements

The present work has received funding from the European Union’s Horizon 2020 Marie Skłodowska Curie Innovative Training Network Greenedge (GA. No. 953775).