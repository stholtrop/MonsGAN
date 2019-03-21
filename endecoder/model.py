from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizer import Adam
from keras import backend as K

class MonsBuilder:
	def __init__(self, shape, layers):
		self.inp = Input(shape=shape)
		x = MaxPooling2D((2,2), padding='same')(Conv2D(layers[0], (3, 3), activation='relu', padding='same')(self.inp))
		for filters in shape[1:]:
			x = MaxPooling2D((2,2), padding='same')(Conv2D(filters, (3, 3), activation='relu', padding='same')(x))
		decoder = Conv2D(layers[-1], (3, 3), activation='relu', padding='same')(x)
		x = UpSampling2D((2, 2))(self.decoder)
		for filters in shape[-2::-1]:
			x = MaxPooling2D((2,2), padding='same')(Conv2D(filters, (3, 3), activation='relu', padding='same')(x))
		self.autoencoder = Model(self.inp, x)
		self.autoencoder.compile(optimizer=Adam(lr=LR), loss='mse')
		self.generator = K.function([decoder.get_input()], x.get_ouput())
		self.seed_shape = decoder.input_shape

	def predict(self, inp):
		return self.generator(inp)
	
	def train(self, data, epochs, batch_size):
		self.autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)


