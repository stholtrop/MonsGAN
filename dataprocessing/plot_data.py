import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# Load data
data = np.load('./npys/N43E005.npy')

# Show image
plt.imshow(data)
plt.show()

# 3D-plot
X, Y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()
