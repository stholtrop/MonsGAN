import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import os

def datalist():
	return map(lambda x: np.load(os.path.join("./npys", x)), os.listdir("./npys"))
def display_set(data, *args):
	for ind, func in enumerate(args):
		ax = plt.subplot(1, len(args), 1 + ind)
		func(data)
	plt.show()

def iterate_data(*args):
	for i in datalist():
		plt.close('all')
		display_set(i, *args)
	
def map2d(data):
	data[data < 0] = 0
	plt.imshow(data)

def shadow_map(data):
	x, y = np.gradient(data)

	slope = np.pi/2 - np.arctan(np.sqrt(x*x + y*y))

	aspect = np.arctan2(-x, y)

	altitude = 5*np.pi/6
	azimuth = np.pi/2

	shaded = np.sin(altitude)*np.sin(slope) + np.cos(altitude) *np.cos(slope)*np.cos((azimuth - np.pi/2) - aspect)

	plt.imshow(shaded, cmap='Greys')

def map3d(data, colour=cm.jet):
	X, Y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, data/22, cmap=colour, linewidth=0, antialiased=False)
	ax.auto_scale_xyz([0, 3601], [0, 3601], [0, 5000/22])

if __name__=='__main__':
	iterate_data(shadow_map, map2d)
