from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
# from keras.optimizer import Adam
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import tensorflow
import numpy as np
import matplotlib.pyplot as plt

class MonsBuilder:
	def __init__(self, shape, layers):
		self.inp = Input(shape=shape)
		x = MaxPooling2D((2,2), padding='same')(Conv2D(layers[0], (3, 3), activation='relu', padding='same')(self.inp))
		for filters in layers[1:]:
			x = MaxPooling2D((2,2), padding='same')(Conv2D(filters, (3, 3), activation='relu', padding='same')(x))
		decoder = Conv2D(layers[-1], (3, 3), activation='relu', padding='same')(x)
		x = UpSampling2D((2, 2))(decoder)
		for filters in layers[-2::-1]:
			x = UpSampling2D((2,2))(Conv2D(filters, (3, 3), activation='relu', padding='same')(x))
		self.autoencoder = Model(self.inp, x)
		self.autoencoder.compile(optimizer="adam", loss='mse')
		self.generator = K.function([decoder], x)
		self.seed_shape = decoder.shape

	def predict(self, inp):
		return self.generator(inp)
	
	def train(self, data, epochs, batch_size):
		data = tensorflow.reshape(data, data.shape + tuple([1]))
		self.autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True, steps_per_epoch=1)


(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

model = MonsBuilder((28, 28, 1), [8])
model.train(x_test, 50, 200)

random_vector = np.random.random(model.seed_shape)
plt.imshow(random_vector)
plt.show()
results = model.predict(random_vector)
plt.imshow(results)
plt.show()
