import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Display an example digit from the dataset
some_digit = x_train[0]
some_digit.shape
plt.imshow(some_digit)
plt.show()

# Introduce noise to the training and test datasets
noise_factor = 0.05
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Display an example digit with added noise
some_noisy_digit = x_train_noisy[0]
plt.imshow(some_noisy_digit)
plt.show()

# Reshape datasets for autoencoder training
x_train_noisy = x_train_noisy.reshape((len(x_train_noisy), np.prod(x_train_noisy.shape[1:])))
x_test_noisy = x_test_noisy.reshape((len(x_test_noisy), np.prod(x_test_noisy.shape[1:])))

# Define the architecture of the autoencoder
input_shape = (784,)
input_layer = Input(shape=input_shape)
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# Create and compile the autoencoder model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder on the noisy datasets
autoencoder.fit(x_train_noisy, x_train_noisy,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_noisy))

# Reconstruct images using the trained autoencoder
decoded_imgs = autoencoder.predict(x_test_noisy)

# Display a reconstructed digit
some_reconstructed_digit = decoded_imgs[0].reshape(28, 28)
plt.imshow(some_reconstructed_digit)
plt.show()

# Evaluate the model on the test set
score = autoencoder.evaluate(x_test_noisy, x_test_noisy, verbose=0)
print('Test loss:', score)
