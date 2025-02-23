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
