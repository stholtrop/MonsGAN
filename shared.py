from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))


input_img = Input(shape=(28*28,))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

encoder = Model(input_img, encoded)

input_enc = Input(shape=(16,))
decoded = Dense(128, activation='relu')(input_enc)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(28*28, activation='relu')(decoded)

decoder = Model(input_enc, decoded)
autoencoder = Model(encoder.inputs, decoder(encoder.outputs))
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train, X_train, epochs=5, batch_size=256, shuffle=True, validation_split=0.2)

decoded_imgs = autoencoder.predict(X_test)
encoded_imgs = encoder.predict(X_test)

test_img = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]).flatten()
print(encoded_imgs[0].shape)
print(test_img.shape)
result = decoder.predict(test_img.reshape(1, 16))
plt.figure(figsize=(20, 4))
for i in range(10):
    # original
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')

    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(result.reshape(28, 28))
plt.subplot(2,1,2)
plt.imshow(test_img.reshape(4,4))
plt.show()