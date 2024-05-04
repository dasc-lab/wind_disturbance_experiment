import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 2)  # Position and velocity of the robot
y = np.sin(2 * np.pi * X[:, 0]) * np.cos(3 * np.pi * X[:, 1])  # Function dependent on position and velocity

# Define kernel with white kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

# Fit Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
gp.fit(X, y)

# Define grid for plotting
x1_min, x1_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
x2_min, x2_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))

# Predict using the fitted model
X_test = np.vstack([xx1.ravel(), xx2.ravel()]).T
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.colorbar(label='Function Value')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Data')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.colorbar(label='Function Value')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Data with GP prediction')
plt.contourf(xx1, xx2, y_pred.reshape(xx1.shape), cmap='viridis', alpha=0.5)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot original data
ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o', label='Data')

# Plot predicted surface
ax.plot_surface(xx1, xx2, y_pred.reshape(xx1.shape), alpha=0.5, cmap='viridis', edgecolor='none')

ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('Function Value')
ax.set_title('Data and GP Prediction in 3D')

plt.show()