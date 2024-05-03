import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Generate synthetic 3D data
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 3)  # 3D points
y = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) * np.sin(2 * np.pi * X[:, 2])  # Synthetic response

# Define the Gaussian Process model with a Radial Basis Function (RBF) kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

# Fit the model to the data
gp.fit(X, y)

# Generate new test points to make predictions
x1_min, x1_max = 0, 1
x2_min, x2_max = 0, 1
x3_min, x3_max = 0, 1
x1, x2, x3 = np.meshgrid(np.linspace(x1_min, x1_max, 20),
                         np.linspace(x2_min, x2_max, 20),
                         np.linspace(x3_min, x3_max, 20))
X_test = np.vstack((x1.flatten(), x2.flatten(), x3.flatten())).T

# Make predictions with the trained model
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', label='Data')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='magma', alpha=0.1, label='Predictions')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.title('Gaussian Process Regression on 3D Data')
plt.legend()
plt.show()
