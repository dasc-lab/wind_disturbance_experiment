from jax import config
config.update("jax_enable_x64", True)
import pickle
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox
import matplotlib.pyplot as plt
import numpy as np
import os,sys
sys.path.append('GPJax')
import gpjax as gpx


gp_models_path = 'gp_models/'
training_path = 'datasets/training/'
test_path = 'datasets/testing/'
factor = 5.0
training_disturbance_path = 'datasets/training/'
training_disturbance_x_path = training_disturbance_path + 'training_disturbance_x.npy'
training_disturbance_y_path = training_disturbance_path + 'training_disturbance_y.npy'
training_disturbance_z_path = training_disturbance_path + 'training_disturbance_z.npy'

training_disturbance_x = np.load(training_disturbance_x_path)
training_disturbance_y = np.load(training_disturbance_y_path)
training_disturbance_z = np.load(training_disturbance_z_path)


training_disturbance = np.column_stack((training_disturbance_x,training_disturbance_y,training_disturbance_z))

models_path = 'gp_models/'
training_input_path = training_path + 'training_input.npy'
training_input = np.load(training_input_path)

counter = 0

for gp_models_path in sorted(os.listdir(models_path)):
    print(gp_models_path)
    if counter == 0:
        with open(models_path + gp_models_path, 'rb') as f:
            assert gp_models_path == 'sparsegp_model_x_norm5_clipped.pkl'
            gp_x = pickle.load(f)
    elif counter == 1:
        with open(models_path + gp_models_path, 'rb') as f:
            assert gp_models_path == 'sparsegp_model_y_norm5_clipped.pkl'
            gp_y = pickle.load(f)
    elif counter == 2:
        with open(models_path + gp_models_path, 'rb') as f:
            assert gp_models_path == 'sparsegp_model_z_norm5_clipped.pkl'
            gp_z = pickle.load(f) 

    counter = counter + 1


test_disturbance_path = 'datasets/testing/'
test_disturbance_path_x = test_disturbance_path + 'test_disturbance_x.npy'
test_disturbance_x = np.load(test_disturbance_path_x)
test_disturbance_path_y = test_disturbance_path + 'test_disturbance_y.npy'
test_disturbance_y = np.load(test_disturbance_path_y)
test_disturbance_path_z = test_disturbance_path + 'test_disturbance_z.npy'
test_disturbance_z = np.load(test_disturbance_path_z)
test_input_path = test_disturbance_path + 'test_input.npy'
test_input = np.load(test_input_path)
test_disturbance = np.column_stack((test_disturbance_x, test_disturbance_y, test_disturbance_z))


D_x = gpx.Dataset(X=training_input, y=training_disturbance_x)
sigma_inv = gp_x.posterior.compute_sigma_inv(D_x)
latent_dist = gp_x.posterior.predict_with_sigma_inv(training_input, D_x, Sigma_inv=sigma_inv)
predictive_dist = gp_x.posterior.likelihood(latent_dist)
pred_mean = predictive_dist.mean().reshape(-1,1)
pred_std = predictive_dist.stddev().reshape(-1,1)

# D_y = gpy.

# predictive_dist = gpx.posterior.likelihood(latent_dist_x)
# pred_mean = predictive_dist.mean()
# pred_std = predictive_dist.stddev()
plt.figure()
plt.plot(jnp.arange(training_disturbance_x.shape[0]), training_disturbance_x, 'r*', markersize=10, label='Actual Data')
plt.plot(jnp.arange(pred_mean.shape[0]), pred_mean, marker = '.', linestyle = '-', color ='b', markersize=10, label='Predictive Mean')
plt.fill_between(jnp.arange(pred_mean.shape[0]), 
                 pred_mean.flatten() - 1.96 * pred_std.flatten(), 
                 pred_mean.flatten() + 1.96 * pred_std.flatten(), 
                 color='orange', alpha=0.2, label='95% Confidence Interval')

# plt.title("Whole Testset")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
# plt.savefig(plot_path+'full_testset.png')
plt.show()
