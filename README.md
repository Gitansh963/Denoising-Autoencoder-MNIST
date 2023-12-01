# Denoising-Autoencoder-MNIST
This Python script uses a denoising autoencoder implemented with `tensorflow` and `keras` to clean noisy images from the MNIST dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The script requires the following Python libraries:

- numpy
- tensorflow
- matplotlib

You can install these libraries using pip:

```bash
pip install numpy tensorflow matplotlib
```

Usage
The script first loads the MNIST dataset and normalizes the images. It then adds noise to the images using a Gaussian distribution.

The noisy images are reshaped and fed into a denoising autoencoder, which is a type of neural network that learns to remove noise from the images.

The autoencoder is trained using the Adam optimizer and mean squared error loss function. After training, the autoencoder is used to predict (denoise) the test images.

The script also includes code to visualize the original, noisy, and denoised images using matplotlib.

Authors
  Gitansh Mittal

