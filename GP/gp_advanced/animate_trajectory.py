import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0,5,5000)
y = np.ones(x.shape)
z = 5* np.ones(x.shape)
# p  = ax.plot(x[0],y[0],z[0], 'r')#,c = 'b', s = 20)
p = ax.scatter( x[0], y[0], z[0], s=60 )
T = x.shape[0]
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
name = 'test_animated_plot.mp4'
writer = FFMpegWriter(fps=12, metadata=metadata)
with writer.saving(fig, name, 100):
    for t in range(T):

        #scatter
        p._offsets3d = ( x[0:t+1],y[0:t+1], z[0:t+1] )

        # plot
        # p[0].set_xdata(x[0:t+1])
        # p[0].set_ydata(y[0:t+1])
        # p[0].set_3d_properties(z[0:t+1])
        writer.grab_frame()
        fig.canvas.draw()
        fig.canvas.flush_events()