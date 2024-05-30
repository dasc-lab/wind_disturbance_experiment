from jax import config

config.update("jax_enable_x64", True)
import pickle
import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox
import matplotlib.pyplot as plt
import numpy as np
home_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/'


with open(home_path+'gpmodels/gp_model_x_norm3_cir2.pkl', 'rb') as f:
    opt_posterior = pickle.load(f)

factor = 3

x = jnp.load(home_path + 'npy_folder/training_input.npy')
y = jnp.load(home_path + 'npy_folder/training_disturbance_x.npy')
disturbance_x = jnp.load(home_path + 'npy_folder/test_disturbance_x.npy')
#wind_disturbance_x = jnp.load(home_path + 'wind_disturbance_x.npy')
plt.figure()
plt.plot(disturbance_x)
plt.show()

plt.figure()
plt.plot(y)
plt.show()
D = gpx.Dataset(X=x, y=y)
xtest = jnp.load(home_path + 'npy_folder/test_input.npy')
#xtest = jnp.load(home_path+ 'fullset_input.npy')
latent_dist = opt_posterior(xtest, D)
predictive_dist = opt_posterior.likelihood(latent_dist)

# Obtain the predictive mean and standard deviation
pred_mean = predictive_dist.mean()
pred_std = predictive_dist.stddev()

pred_mean = pred_mean*factor
pred_std = pred_std*factor
disturbance_x = disturbance_x*factor
# latent_dist = opt_posterior(xtest, D)
# predictive_dist = opt_posterior.likelihood(latent_dist)
# pred_mean = predictive_dist.mean()
# pred_std = predictive_dist.stddev()

#plt.scatter(jnp.arange(disturbance_x.shape[0]), disturbance_x*factor, color='blue', label='Actual Data', alpha=0.5)
plt.scatter(jnp.arange(disturbance_x.shape[0]), disturbance_x, color='blue', label='Actual Data', alpha=0.5)
plt.plot(jnp.arange(pred_mean.shape[0]), pred_mean, color='red', label='Predictive Mean')
plt.fill_between(jnp.arange(pred_mean.shape[0]), 
                 pred_mean.flatten() - 1.96 * pred_std.flatten(), 
                 pred_mean.flatten() + 1.96 * pred_std.flatten(), 
                 color='orange', alpha=0.2, label='95% Confidence Interval')

plt.title("Gaussian Process Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# num_indices = int(xtest.shape[0]/3)
# plot_indices = np.random.choice(pred_mean.shape[0], num_indices, replace=False)
# print("plot_indices shape is: ", plot_indices.shape)
# sorted_indices = np.sort(plot_indices)

# pred_mean = pred_mean[sorted_indices]
# pred_std = pred_std[sorted_indices]
# actual_output = disturbance_x[sorted_indices]

pred_mean = pred_mean[::4]
pred_std = pred_std[::4]
actual_output = disturbance_x[::4]
plt.figure()
plt.plot(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), actual_output, 'r*', markersize=10, label='Actual Data')
plt.plot(np.linspace(0,pred_mean.shape[0],pred_mean.shape[0]), pred_mean, marker = '.', linestyle = '-', color ='b', markersize=10, label='GP Prediction')

#axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], 'r*', markersize=10, label='Actual Data')
#axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], marker = '*',linestyle = '-', color = 'r', markersize=10, label='Actual Data')

plt.fill_between(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), 
                    (pred_mean - 1.96 * pred_std).flatten(), 
                    (pred_mean + 1.96 * pred_std).flatten(), 
                    color='orange', alpha=0.2, label='95% Confidence Interval')
plt.show()