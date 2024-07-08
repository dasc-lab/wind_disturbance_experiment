import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Generate synthetic 3D data
wind_disturbance = np.load("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/disturbance.npy")
input = np.load("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/input_to_gp.npy")
cutoff = 4000
threshold = 1000
set_size = cutoff - threshold
# print("wind disturbance shape", wind_disturbance.shape)
# print("input shape", input.shape)
wind_disturbance = wind_disturbance[threshold:cutoff,:]
input = input[threshold:cutoff,:]
print("wind disturbance shape", wind_disturbance.shape)
print("input shape", input.shape)
training_size = n = 200
wind_disturbance_x = np.array(wind_disturbance[:,1]).reshape(-1,1)
indices = np.random.choice(wind_disturbance_x.size, (training_size,) , replace=False)
X = input[indices]
y = wind_disturbance_x[indices]
# np.random.seed(0)
# n_samples = 100
# X = np.random.rand(n_samples, 3)  # 3D points
# y = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) * np.sin(2 * np.pi * X[:, 2])  # Synthetic response

# Define the Gaussian Process model with a Radial Basis Function (RBF) kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

# Fit the model to the data
gp.fit(X, y)

# Generate new test points to make predictions
# x1_min, x1_max = 0, 1
# x2_min, x2_max = 0, 1
# x3_min, x3_max = 0, 1
# x1, x2, x3 = np.meshgrid(np.linspace(x1_min, x1_max, 20),
#                          np.linspace(x2_min, x2_max, 20),
#                          np.linspace(x3_min, x3_max, 20))
# X_test = np.vstack((x1.flatten(), x2.flatten(), x3.flatten())).T
X_test = input
# Make predictions with the trained model
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', label='Data')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='magma', alpha=0.1, label='Predictions')
ax.set_zlim(-0.2,-0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Gaussian Process Regression on 3D Data')
plt.legend()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust subplot grid as needed
axes = axes.flatten()
plot_size = plot_n = 50
plot_indices = np.random.choice( set_size, (plot_n,) , replace=False)
xtest = X_test
pred_mean = y_pred
pred_std = sigma
dim = 6 # or 1
print(pred_mean-2*pred_std)
print(pred_mean+2*pred_std)
for i in range(dim):
    axes[i].plot(xtest[:, i][plot_indices], pred_mean[plot_indices], 'b.', markersize=10, label='GP Prediction')
    axes[i].plot(input[:,i][plot_indices], wind_disturbance_x[plot_indices], 'r.', markersize=10, label='Actual Data')
    # axes[i].plot(x[:,i][plot_indices], y[plot_indices], 'r.', markersize=10, label='Actual Data')
    # print(pred_mean)
    
    axes[i].fill_between(xtest[:, i], 
                         (pred_mean - 1.96 * pred_std), 
                         (pred_mean + 1.96 * pred_std), 
                         color='orange', alpha=0.2, label='95% Confidence Interval')
    
    axes[i].set_title(f'Dimension {i+1}')
    axes[i].set_xlabel(f'Input {i+1}')
    #axes[i].set_title(f'Disturbance vs {dim_d[i]}' )
    #axes[i].set_xlabel(dim_d[i])
    axes[i].set_ylabel('Disturbance (m/s^2)')
    axes[i].legend()

fig.tight_layout()
plt.savefig("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/GP_plots.png")
plt.show()

