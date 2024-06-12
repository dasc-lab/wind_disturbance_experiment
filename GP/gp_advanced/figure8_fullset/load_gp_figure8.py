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


dataset_path = home_path + 'figure8_fullset/'
model_path = dataset_path + 'models/'
npy_data_path = dataset_path + 'npy_data_folder/'
plot_path = dataset_path + 'test_plots/'
gp_x_file = 'gp_model_x_norm5_figure8.pkl'
gp_y_file = 'gp_model_y_norm5_figure8.pkl'
gp_z_file = 'gp_model_z_norm5_circle.pkl'
with open(model_path+gp_x_file, 'rb') as f:
    opt_posterior = pickle.load(f)

factor = 3

x = jnp.load(npy_data_path + 'training_input.npy')
y = jnp.load(npy_data_path + 'training_disturbance_x.npy')
disturbance_x = jnp.load(npy_data_path + 'test_disturbance_x.npy')
#wind_disturbance_x = jnp.load(home_path + 'wind_disturbance_x.npy')
# plt.figure()
# plt.plot(disturbance_x)
# plt.show()

# plt.figure()
# plt.plot(y)
# plt.show()
D = gpx.Dataset(X=x, y=y)



########################################################################
######################## Plot Test dataset #############################
########################################################################
xtest = jnp.load(npy_data_path + 'test_input.npy')
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
# plt.scatter(jnp.arange(disturbance_x.shape[0]), disturbance_x, color='blue', label='Actual Data', alpha=0.5)
# plt.plot(jnp.arange(pred_mean.shape[0]), pred_mean, color='red', label='Predictive Mean')
plt.plot(jnp.arange(disturbance_x.shape[0]), disturbance_x, 'r*', markersize=10, label='Actual Data')
plt.plot(jnp.arange(pred_mean.shape[0]), pred_mean, marker = '.', linestyle = '-', color ='b', markersize=10, label='Predictive Mean')
plt.fill_between(jnp.arange(pred_mean.shape[0]), 
                 pred_mean.flatten() - 1.96 * pred_std.flatten(), 
                 pred_mean.flatten() + 1.96 * pred_std.flatten(), 
                 color='orange', alpha=0.2, label='95% Confidence Interval')

plt.title("Whole Testset")
plt.xlabel("time")
plt.ylabel("disturbance")
plt.legend()
plt.savefig(plot_path+'full_testset.png')
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
plt.title("Gaussian Process Regression Sparse Testset")
plt.savefig(plot_path+'sparse_testset.png')
plt.show()


########################################################################
################## Plot Sparse Traning dataset #########################
########################################################################


latent_dist = opt_posterior(x, D)
predictive_dist = opt_posterior.likelihood(latent_dist)

# Obtain the predictive mean and standard deviation
pred_mean = predictive_dist.mean()
pred_std = predictive_dist.stddev()

pred_mean = pred_mean*factor
pred_std = pred_std*factor
y = y*factor
slice = 5
pred_mean = pred_mean[::slice]
pred_std = pred_std[::slice]
actual_output = y[::slice]
plt.figure()
plt.plot(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), actual_output, 'r*', markersize=10, label='Actual Data')
plt.plot(np.linspace(0,pred_mean.shape[0],pred_mean.shape[0]), pred_mean, marker = '.', linestyle = '-', color ='b', markersize=10, label='GP Prediction')

#axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], 'r*', markersize=10, label='Actual Data')
#axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], marker = '*',linestyle = '-', color = 'r', markersize=10, label='Actual Data')

plt.fill_between(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), 
                    (pred_mean - 1.96 * pred_std).flatten(), 
                    (pred_mean + 1.96 * pred_std).flatten(), 
                    color='orange', alpha=0.2, label='95% Confidence Interval')
plt.title("Gaussian Process Regression Sparse Training Set")
plt.savefig(plot_path+'sparse_trainingset.png')
plt.show()

