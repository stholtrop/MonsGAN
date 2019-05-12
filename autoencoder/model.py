from keras.layers import Dense, Input, MaxPool1D
from keras.models import Model
import os
import numpy as np
import matplotlib.pyplot as plt


class Mons:
    def __init__(self, input_size, compression):
        encoded = self.input1 = Input(shape=(input_size,))
        encoded = Dense(compression)(encoded)
        self.encoder = Model(self.input1, encoded)
        self.encoder.summary()
        decoded = self.input2 = Input(shape=(compression,))
        decoded = Dense(input_size)(decoded)
        self.decoder = Model(self.input2, decoded)
        self.decoder.summary()
        self.autoencoder = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))
        self.autoencoder.compile(optimizer="adam", loss="mse")
        self.autoencoder.summary()
        self.data = None
        self.blocks = []

    @staticmethod
    def datalist():
        return list(map(lambda x: np.load(os.path.join("../dataprocessing/npys", x)), os.listdir("../dataprocessing/npys")))

    def split(self, block, n):
        length = block.shape[0] // n
        blocks = []
        for i in range(n):
            for j in range(n):
                blocks.append(block[i*length:(i+1)*length, j*length:(j+1)*length].flatten())
        return blocks

    def load_data(self):
        if not self.data:
            self.data = np.array(self.datalist())
            # Split data
            for data_block in self.data:
                self.blocks += self.split(data_block, 10)
            self.blocks = np.array(self.blocks)
            print(self.blocks.shape)

    def train(self):
        self.load_data()
        self.autoencoder.fit(self.blocks, self.blocks, epochs=5)


if __name__ == "__main__":
    network = Mons(360**2, 100)
    network.train()
    # Test
    test_data = network.blocks[0:10]
    result_data = network.autoencoder.predict(test_data)
    plt.figure(figsize=(20, 4))
    for i in range(10):
        # original
        plt.subplot(2, 10, i + 1)
        plt.imshow(test_data[i].reshape(360, 360))
        plt.gray()
        plt.axis('off')

        # reconstruction
        plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(result_data[i].reshape(360, 360))
        plt.gray()
        plt.axis('off')

    plt.tight_layout()
    plt.show()

