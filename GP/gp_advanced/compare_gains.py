import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ideal_array = np.load('compare_trajectory/ideal_trajectory.npy')
unoptimized_array = np.load('compare_trajectory/unoptimized_trajectory_cir_7_4.npy')
optimized_array = np.load('compare_trajectory/optimized_trajectory_cir_14_7.npy')
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
# ax.set_xlim(-0.8, 0.8)
# ax.set_ylim(0.4,1.1)
ax.set_zlim(0,1)
ax.scatter(x_ideal, y_ideal, z_ideal, c = 'b', label = 'ideal trajectory')
ax.scatter(x, y, z, c= 'r', label = 'unoptimized trajectory')
ax.scatter(x_op, y_op, z_op, c= 'g', label = 'optimized trajectory')
ax.legend()
# plt.savefig("compare_trajectory/trajectory_comp_cir_r0.4_w1.0_c0.60_h0.5.png")
# plt.show()

ideal_array = np.load('compare_trajectory/ideal_trajectory_figure8_r0.4_w1_c00_h0.5.npy')
unoptimized_array = np.load('compare_trajectory/unoptimized_trajectory_figure8_r0.4_w1_c00_h0.5.npy')
optimized_array = np.load('compare_trajectory/optimized_trajectory_figure8_14_7_r0.4_w1_c00_h0.5.npy')
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
# ax.set_xlim(-0.8, 0.8)
# ax.set_ylim(0.4,1.1)
ax.set_zlim(0,1)
ax.scatter(x_ideal, y_ideal, z_ideal, c = 'b', label = 'ideal trajectory')
ax.scatter(x_op, y_op, z_op, c= 'g', label = 'optimized trajectory')
ax.scatter(x, y, z, c= 'r', label = 'unoptimized trajectory')

ax.legend()
# plt.savefig("compare_trajectory/trajectory_comp_cir_r0.4_w1.0_c0.60_h0.5.png")
# plt.show()

ideal_array = np.load('compare_trajectory/ideal_trajectory_figure8_r0.4_w1_c080_h0.5_kxv7_00_4_00.npy')
unoptimized_array = np.load('compare_trajectory/unoptimized_trajectory_figure8_r0.4_w1_c080_h0.5_kxv7_00_4_00.npy')
optimized_array = np.load('compare_trajectory/optimized_trajectory_figure8_15_7_r0.4_w1_c080_h0.5_kxv7_00_4_00.npy')
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
# ax.set_xlim(-0.8, 0.8)
# ax.set_ylim(0.4,1.1)
ax.set_zlim(0,1)
ax.scatter(x_ideal, y_ideal, z_ideal, c = 'b', label = 'ideal trajectory')
ax.scatter(x_op, y_op, z_op, c= 'g', label = 'optimized trajectory')
ax.scatter(x, y, z, c= 'r', label = 'unoptimized trajectory')

ax.legend()
# plt.show()

ideal_array = np.load('compare_trajectory/ideal_trajectory_cir_r0.4_w1.0_c0.00_h0.5.npy')
unoptimized_array = np.load('compare_trajectory/unoptimized_trajectory_cir_r0.4_w1.0_c0.00_h0.5_kxv7_00_4_00.npy')
optimized_array = np.load('compare_trajectory/optimized_trajectory_cir_r0.4_w1.0_c0.00_h0.5_kxv19_9.npy')
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
# ax.set_xlim(-0.8, 0.8)
# ax.set_ylim(0.4,1.1)
ax.set_zlim(0,1)
ax.scatter(x_ideal, y_ideal, z_ideal, c = 'b', label = 'ideal trajectory')
ax.scatter(x_op, y_op, z_op, c= 'g', label = 'optimized trajectory')
ax.scatter(x, y, z, c= 'r', label = 'unoptimized trajectory')

ax.legend()
plt.show()