# Continuous Wavelet Transform Signal Generation 🌊

**Work in Progress**: Exploring CWT for signal generation using deep learning approaches.

---

## Overview

This project investigates **Continuous Wavelet Transform (CWT)** for generating signals from other signals using two different deep learning architectures:

- **Encoder-Decoder** (VAE-based approach)
- **Diffusion Models** (Latent diffusion)

## Approach

**Pipeline:**
1. **CWT Transformation** - Convert input signals to time-frequency representations
2. **Latent Encoding** - Encode CWT spectrograms into latent space 
3. **Signal Generation** - Generate new signals using:
   - VAE Encoder-Decoder architecture
   - Latent Diffusion Models

## Current Status

🚧 **Work in Progress** - Experimental exploration phase

**Files:**
- `models.py` - Encoder/Decoder and Diffusion model architectures
- `cwt_utils.py` - CWT processing utilities 
- `train.py` - Training pipeline (in development)
- `dataset.py` - Data handling for signal processing

## Technologies

- **PyTorch** - Deep learning framework
- **PyWavelets** - CWT implementation
- **NumPy/Matplotlib** - Signal processing and visualization

---

**Note:** This is an experimental project exploring the intersection of wavelet transforms and generative models for signal synthesis.