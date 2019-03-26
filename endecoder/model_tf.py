import tensorflow.contrib.keras as kr
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class MonsBuilder:
    def __init__(self, shape: Tuple, filters_per_layer: Tuple):
        self.input = kr.layers.Input(shape=shape)
        # Create convolutional layers
        next_layer = kr.layers.MaxPooling2D((2, 2), padding='same')(kr.layers.Conv2D(filters_per_layer[0], (3, 3), activation='relu', padding='same')(self.input))
        for n_filters in filters_per_layer[1:-1]:
            next_layer = kr.layers.MaxPooling2D((2, 2), padding='same')(kr.layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(next_layer))
        # Decoder input and encoder output
        self.decoder_input = kr.layers.MaxPooling2D((2, 2), padding='same')(kr.layers.Conv2D(filters_per_layer[-1], (3, 3), activation='relu', padding='same')(next_layer))
        # Start upsampling
        next_layer = kr.layers.UpSampling2D((2, 2))(self.decoder_input)
        for filters in filters_per_layer[:-2:-1]:
            next_layer = kr.layers.MaxPooling2D((2, 2), padding='same')(kr.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(next_layer))
        self.last_layer = next_layer
        # Autoencoder
        self.autoencoder = kr.models.Model(self.input, self.last_layer)
        self.autoencoder.compile(optimizer="adam", loss="mse")
        # Encoder
        self.encoder = kr.models.Model(self.input, self.decoder_input)
        # Decoder
        self.decoder = kr.models.Model(self.decoder_input, self.last_layer)
        self.compressed_size = self.decoder_input.shape

    def train_autoencoder(self, data: np.ndarray, **args):
        self.autoencoder.train(data, data, **args)

    def encode(self, data: np.ndarray):
        return self.encoder.predict(data)

    def decode(self, data: np.ndarray):
        return self.decoder.predict(data)


# Main test program
if __name__ == "__main__":
    (x_train, _), (x_test, _) = kr.datasets.mnist.load_data()
    model = MonsBuilder((28, 28, 1), (16, 8, 8))
    print(model.compressed_size)
    model.train_autoencoder(x_train, epochs=20, batch_size=256, shuffle=True, validation_data=(x_test, x_test), verbose=True)
    random_vector = np.random.random(model.compressed_size)
    plt.imshow(random_vector)
    plt.show()
    results = model.decode(random_vector)
    plt.imshow(results)
    plt.show()
