# Kaira Metrics Module

This module provides metrics for evaluating the performance of communication systems and signal processing algorithms.

## Overview

The metrics module in Kaira is designed with the following features:

- **Modular Structure**: Metrics are organized by category (image, signal, etc.)
- **Registry System**: Metrics can be registered and accessed by name
- **Factory Functions**: Create commonly used combinations of metrics
- **Consistent Interface**: All metrics inherit from a common base class

## Available Metrics

### Image Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise
- **SSIM (Structural Similarity Index Measure)**: Measures the perceptual similarity between two images
- **MS-SSIM (Multi-Scale SSIM)**: An extension of SSIM that considers multiple scales
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual similarity using deep features

### Signal Metrics

- **SNR (Signal-to-Noise Ratio)**: Compares the level of the desired signal to the level of background noise
- **BER (Bit Error Rate)**: Measures the number of bit errors divided by the total number of bits transmitted

## Usage Examples

### Basic Usage

