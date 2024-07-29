import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,0.5)
ax.set_zlim(0,1)
ideal_array = np.load('compare_trajectory/ideal_trajectory_cir_r0.4_w1.0_c0.00_h0.5.npy')
unoptimized_array = np.load('compare_trajectory/unoptimized_trajectory_cir_r0.4_w1.0_c0.00_h0.5_kxv7_00_4_00.npy')
optimized_array = np.load('compare_trajectory/optimized_trajectory_cir_r0.4_w1.0_c0.00_h0.5_kxv19_9.npy')
# x = np.linspace(0,5,5000)
# y = np.ones(x.shape)
# z = 5* np.ones(x.shape)
x_ideal = ideal_array[:,0]
y_ideal = ideal_array[:,1] 
z_ideal = ideal_array[:,2]



x = unoptimized_array[:,0]
y = unoptimized_array[:,1]
z = unoptimized_array[:,2]

x_op = optimized_array[:,0]
y_op = optimized_array[:,1]
z_op = optimized_array[:,2]
# p  = ax.plot(x[0],y[0],z[0], 'r')#,c = 'b', s = 20)
ideal_trajectory = ax.scatter(x_ideal[0], y_ideal[0], z_ideal[0], c='black', s= 20, marker = '^', label = 'reference trajectory')
optimized_trajectory = ax.scatter(x_op[0], y_op[0], z_op[0], c='springgreen', s= 20, label = 'optimized trajectory')
trajectory = ax.scatter( x[0], y[0], z[0], c = 'brown', s=20, alpha = '0.5', label = 'unoptimized trajecotry')
ax.legend()
T = x.shape[0]
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
name = 'cir_r0.4_w1.0_c0.00_h0.5.mp4'
writer = FFMpegWriter(fps=20, metadata=metadata)
with writer.saving(fig, 'animated_plots/'+name, 100):
    for t in range(T):

        #scatter
        trajectory._offsets3d = ( x[0:t+1],y[0:t+1], z[0:t+1] )
        ideal_trajectory._offsets3d = (x_ideal[0:t+1], y_ideal[0:t+1], z_ideal[0:t+1] )
        optimized_trajectory._offsets3d = (x_op[0:t+1], y_op[0:t+1], z_op[0:t+1] )
        # plot
        # p[0].set_xdata(x[0:t+1])
        # p[0].set_ydata(y[0:t+1])
        # p[0].set_3d_properties(z[0:t+1])
        writer.grab_frame()
        fig.canvas.draw()
        fig.canvas.flush_events()