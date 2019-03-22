import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import os

def datalist():
	return map(lambda x: np.load(os.path.join("./npys", x)), os.listdir("./npys"))

def iterate_data(display_func):
	for i in datalist():
		display_func(i)	

def map2d(data):
	data[data < 0] = 0
	plt.imshow(data)
	plt.show()

def map3d(data, colour=cm.jet):
	X, Y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, data/22, cmap=colour, linewidth=0, antialiased=False)
	ax.auto_scale_xyz([0, 3601], [0, 3601], [0, 5000/22])
	plt.show()

if __name__=='__main__':
	iterate_data(map2d)
