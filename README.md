<p align="center">
  <img src="docs/logo.png" />
</p>

# Kaira -- Wireless simulation toolbox for PyTorch


Kaira is a toolbox for simulating wireless communication systems, focused on wireless data transmission using deep joint source-channel coding (DeepJSCC). It is built on top of PyTorch and provides a simple and flexible framework for deep learning-based image transmission. Kaira is designed to accelerate research in wireless communication systems and facilitate the development of new algorithms. It is also easy to use and can be integrated with existing PyTorch projects.

### DeepJSCC
Deep joint source-channel coding (DeepJSCC) is a wireless data transmission method that uses deep neural networks to directly map data to channel input symbols, bypassing the need for explicit compression or error correction codes. This end-to-end framework employs two convolutional neural networks as encoder and decoder, functioning like an autoencoder with a non-trainable layer representing the noisy communication channel. Deep JSCC surpasses traditional digital transmission methods in low signal-to-noise ratio and bandwidth conditions, and gracefully degrades performance as the channel signal-to-noise ratio changes. It can also learn to resist noise and outperform traditional digital communication in slow Rayleigh fading channels.

### Features

1. **Research-Oriented**: Designed to accelerate deep joint-source channel coding research.
2. **Versatility**: Compatible with various data types and neural network architectures.
3. **Ease of Use**: User-friendly and easy to integrate with existing PyTorch projects.
4. **Open Source**: Allows for community contributions and improvements.
5. **Well Documented**: Comes with comprehensive documentation for easy understanding.

### Installation

To install the latest version of kaira, please run the following command:

```bash
pip install kaira
```

### Usage

To use kaira in your project, import it as follows:

```python
import kaira
```

### Examples

We have provided several examples in the `examples/` directory to help you get started.

### Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to contribute.

### Versioning
We use `Semantic versioning <http://semver.org/>`_ for this project.

### License

This project is licensed under the MIT License. See `LICENSE.md` for more details.

### Contact

If you have any questions or feedback, please feel free to open an issue or submit a pull request.