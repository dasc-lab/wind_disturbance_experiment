import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ideal_array = np.load('compare_trajectory/ideal_trajectory.npy')
unoptimized_array = np.load('compare_trajectory/unoptimized_trajectory.npy')
optimized_array = np.load('compare_trajectory/optimized_trajectory.npy')
x_ideal = ideal_array[:,0]
y_ideal = ideal_array[:,1] 
z_ideal = ideal_array[:,2]



x = unoptimized_array[:,0]
y = unoptimized_array[:,1]
z = unoptimized_array[:,2]

x_op = optimized_array[:,0]
y_op = optimized_array[:,1]
z_op = optimized_array[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0,1)
ax.scatter(x_ideal, y_ideal, z_ideal, c = 'b', label = 'ideal trajectory')
ax.scatter(x, y, z, c= 'r', label = 'unoptimized trajectory')
ax.scatter(x_op, y_op, z_op, c= 'g', label = 'optimized trajectory')
ax.legend()
#plt.savefig("trajectory.png")
plt.show()