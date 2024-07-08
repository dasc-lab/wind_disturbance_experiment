from jax import config

config.update("jax_enable_x64", True)
import sys
import os
plotter_path = os.path.join('/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/')
sys.path.append(plotter_path)
from plot_trajectory_ref import cutoff, threshold
from gpjax.base import static_field
import pickle
home_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced'
import tensorflow_probability.substrates.jax.bijectors as tfb

import numpy as np
import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox
import flax
from flax import linen as nn
from gpjax.kernels import SumKernel, White, RBF, Matern32, RationalQuadratic, Periodic, ProductKernel
import matplotlib.pyplot as plt
from gpjax.kernels import AbstractKernel 
from dataclasses import dataclass, field
from typing import Any
from jaxtyping import Float, Array, install_import_hook
import jax
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base import param_field
    import gpjax.kernels as jk
    from gpjax.kernels import DenseKernelComputation
    from gpjax.kernels.base import AbstractKernel
    from gpjax.kernels.computations import AbstractKernelComputation

key = jr.key(123)






def mean_squared_error(y_true, y_pred):
    y_true = jnp.array(y_true)
    y_pred = jnp.array(y_pred)
    return jnp.mean((y_true - y_pred) ** 2)

def read_float_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            value = float(file.read().strip())
        return value
    except FileNotFoundError:
        return None

def write_float_to_file(file_path, value):
    with open(file_path, 'w') as file:
        file.write(f'{value}')

def save_gp_params(file_path, params):
    return

def compare_and_update(file_path, new_value):
    stored_value = read_float_from_file(file_path)

    if stored_value is None or new_value < stored_value:
        write_float_to_file(file_path, new_value)
        print(f'NEW BEST MODEL FOUND: {new_value}')

    else:
        print("new value is ", new_value)
        print(f'Kept the existing value: {stored_value}')

# Path to the file
file_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/training/min_error.txt'



################################### Declare Parameters ##########################################

error_x = error_y = error_z = curr_min_x = curr_min_y = curr_min_z =  0




dim_d = {
    0: "x (m)",
    1: "y (m)",
    2: "z (m)",
    3: "Vx (m/s)",
    4: "Vy (m/s)",
    5: "Vz (m/s)"
}

disturbance_d = {
    0: 'x',
    1: 'y',
    2: 'z'
}

factor = 3
gp_model_x = gp_model_y = gp_model_z = None
pred_mean_x = pred_std_x = pred_mean_y = pred_std_y = pred_mean_z = pred_std_z = None

dim = 6 ## 6 input dims x,y,z,vx,vy,vz
################################### Data Prep ##########################################
disturbance_file_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/disturbance_1.npy'
input_file_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/input_1.npy'
wind_disturbance = jnp.load(disturbance_file_path)


wind_disturbance = wind_disturbance/factor


input = jnp.load(input_file_path)

assert wind_disturbance.shape[0] == input.shape[0]
#cutoff = wind_disturbance.shape[0]-3000
#threshold = 300

# set_size = cutoff - threshold
cutoff = wind_disturbance.shape[0]
threshold = 0
wind_disturbance = wind_disturbance[threshold:cutoff,:]
input = input[threshold:cutoff,:]
wind_disturbance = wind_disturbance[::4]
input = input[::4]

##################### Remove Outliers #####################
outlier_threshold = 5.5
rows_to_remove = jnp.any(wind_disturbance > outlier_threshold, axis=1)
num_outlier = np.sum(rows_to_remove)

print(f"removing {num_outlier} from disturbance, whose shape is {wind_disturbance.shape}")
wind_disturbance = wind_disturbance[~rows_to_remove]
input = input[~rows_to_remove]
print(f"The remaining length is {wind_disturbance.shape}")

np.save("fullset_input.npy", input)
assert wind_disturbance.shape[0] == input.shape[0]

wind_disturbance_x = jnp.array(wind_disturbance[:,0]).reshape(-1,1)
wind_disturbance_y = jnp.array(wind_disturbance[:,1]).reshape(-1,1)
wind_disturbance_z = jnp.array(wind_disturbance[:,2]).reshape(-1,1)
np.save("wind_disturbance_x.npy", wind_disturbance_x)
assert wind_disturbance_x.shape == wind_disturbance_y.shape == wind_disturbance_z.shape# == (set_size,1)

max_acc = 5
count_x = jnp.sum(jnp.abs(wind_disturbance_x)>max_acc)
count_y = jnp.sum(jnp.abs(wind_disturbance_y)>max_acc)
count_z = jnp.sum(jnp.abs(wind_disturbance_z)>max_acc)
print(f"In the entire dataset, for y disturbance, out of {wind_disturbance_y.shape[0]} data points, the absolute values of {count_y} are bigger than 5")
print(f"In the entire dataset, for z disturbance, out of {wind_disturbance_z.shape[0]} data points, the absolute values of {count_z} are bigger than 5")
print(f"In the entire dataset, for x disturbance, out of {wind_disturbance_x.shape[0]} data points, the absolute values of {count_x} are bigger than 5")
training_size = n = int(wind_disturbance.shape[0]*0.8)
training_indices = jr.choice(key, wind_disturbance_x.size, (training_size,) , replace=False)
mask = jnp.ones(wind_disturbance.shape[0], dtype=bool)
mask = mask.at[training_indices].set(False)
test_input = input[mask]
test_disturbance_x = wind_disturbance_x[mask]
test_disturbance_y = wind_disturbance_y[mask]
test_disturbance_z = wind_disturbance_z[mask]
jnp.save('test_input.npy', test_input)
jnp.save('test_disturbance_x.npy', test_disturbance_x)
jnp.save('test_disturbance_y.npy', test_disturbance_y)
jnp.save('test_disturbance_z.npy', test_disturbance_z)
print(f"training on {training_size} datapoints")
training_input = input[training_indices]

np.save("training_input.npy", training_input)
training_disturbance_x = wind_disturbance_x[training_indices]
training_disturbance_y = wind_disturbance_y[training_indices]
training_disturbance_z = wind_disturbance_z[training_indices]
np.save("training_disturbance_x.npy", training_disturbance_x)
np.save("training_disturbance_y.npy", training_disturbance_y)
np.save("training_disturbance_z.npy", training_disturbance_z)
disturbance_x_mean = 0.0
disturbance_y_mean = 0.0
disturbance_z_mean = 0.0
disturbance_x_median = 0.0
disturbance_y_median = 0.0
disturbance_z_median = 0.0
fig, axes = plt.subplots(3, 1, figsize=(15, 10))  # Adjust subplot grid as needed
axes = axes.flatten()
test_set_size = input.shape[0]
plot_size = plot_n = 100
for j in range(3):
    if j == 0:
        wind_disturbance_curr = wind_disturbance_x
    elif j == 1:
        wind_disturbance_curr = wind_disturbance_y
    elif j == 2:
        wind_disturbance_curr = wind_disturbance_z
    else:
        print("shouldn't reach this statement")
    x = input[training_indices]
    y = wind_disturbance_curr[training_indices]
    

    ########################################## GP ##########################################


    D = gpx.Dataset(X=x, y=y)
    noise_level = 0.3
    # Construct the prior
    meanf = gpx.mean_functions.Zero()
    white_kernel = White(variance=noise_level)
    # kernel = SumKernel(kernels=[RBF(), Matern32(), white_kernel])
    rbf_kernel = RBF(active_dims=[0,1,2,3,4,5])
    rational_quadratic_kernel = RationalQuadratic()
    
    periodic_kernel = Periodic()
    composite_kernel = ProductKernel(kernels=[periodic_kernel, rational_quadratic_kernel])
    combined_kernel = SumKernel(kernels=[composite_kernel, rbf_kernel, white_kernel])
    # composite_kernel = ProductKernel(kernels=[periodic_kernel, rbf_kernel])
    # combined_kernel = SumKernel(kernels=[composite_kernel, rational_quadratic_kernel, white_kernel])
    kernel = combined_kernel
   

    prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)

    # Define a likelihood
    likelihood = gpx.likelihoods.Gaussian(num_datapoints = n)

    # Construct the posterior
    posterior = prior * likelihood

    # Define an optimiser
    #optimiser = ox.adam(learning_rate=1e-2)
    optimiser = ox.adam(learning_rate=5e-3) 
    # Define the marginal log-likelihood
    negative_mll = jit(gpx.objectives.ConjugateMLL(negative=True))

    # Obtain Type 2 MLEs of the hyperparameters
    opt_posterior, history =gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=D,
        optim=optimiser,
        num_iters=800,
        safe=True,
        key=key,
    )
    
    print("after training, predicting")
    # Infer the predictive posterior distribution
    xplot = input
    # actual_output = np.delete(wind_disturbance_curr,training_indices)
    # xtest = np.delete(input, training_indices, axis=0)
    actual_output = wind_disturbance_curr
    xtest = input
    assert(actual_output.shape[0] == xtest.shape[0])
    print("input shape = ", input.shape)
    print("test shape = ", xtest.shape)
    
    gp_model_file_path = ''
    if j ==0:
        gp_model_x = opt_posterior
        #gp_model_file_path = home_path + 'gpmodels/gp_model_x_norm3.pkl'
        gp_model_file_path = 'gp_model_x_norm3.pkl'
    if j ==1:
        gp_model_y = opt_posterior
        #gp_model_file_path = home_path + 'gpmodels/gp_model_y_norm3.pkl'
        gp_model_file_path = 'gp_model_y_norm3.pkl'
    if j == 2:
        gp_model_z = opt_posterior
        gp_model_file_path = 'gp_model_z_norm3.pkl'
    with open(gp_model_file_path, 'wb') as file:
        pickle.dump(opt_posterior, file)
    ################# Predicting ###################
    latent_dist = opt_posterior(xtest, D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    # Obtain the predictive mean and standard deviation
    pred_mean = predictive_dist.mean()
    pred_std = predictive_dist.stddev()

    if j == 0:
        pred_mean_x = pred_mean
        pred_std_x = pred_std
        disturbance_x_mean = jnp.mean(pred_mean)
        disturbance_x_median = jnp.median(pred_mean)
    elif j == 1:
        pred_mean_y = pred_mean
        pred_std_y = pred_std
        disturbance_y_mean = jnp.mean(pred_mean)
        disturbance_y_median = jnp.median(pred_mean)
    elif j == 2:
        pred_mean_z = pred_mean
        pred_std_z = pred_std
        disturbance_z_mean = jnp.mean(pred_mean)
        disturbance_z_median = jnp.median(pred_mean)
    else:
        print("shouldn't reach this statement")
        assert j<3
    # traj_dist = opt_posterior(xplot,D)
    # traj_pred = opt_posterior.likelihood(traj_dist)
    # mean = traj_pred.mean()
    # std = traj_pred.stddev()

########################################### Compute Mean Squared Error ##########################################
    error = mean_squared_error(pred_mean,wind_disturbance_curr)
    if j == 0:
        error_x = error
    elif j == 1:
        error_y = error
    elif j == 2:
        error_z = error
    

########################################### Plotting Trajectory ##########################################

    # fig, axes = plt.subplots(3, 1, figsize=(15, 10))  # Adjust subplot grid as needed
    # axes = axes.flatten()
    # test_set_size = xtest.shape[0]
    # plot_size = plot_n = 100
    #plot_indices = jr.choice(key, test_set_size, (plot_n,) , replace=False)
    
    # for i in range(1):
    #     # sorted_indices = jnp.argsort(xtest[:, i][plot_indices])
        
    #     # x_sorted = xtest[:, i][plot_indices][sorted_indices]
    #     # pred_mean_sorted = pred_mean[plot_indices][sorted_indices]
    #     # pred_std_sorted = pred_std[plot_indices][sorted_indices]
        
    #     #axes[i].plot(x_sorted, pred_mean_sorted, 'b.', markersize=10, label='GP Prediction')
    #     axes[i].plot(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), actual_output, 'r*', markersize=10, label='Actual Data')
    #     axes[i].plot(np.linspace(0,xplot.shape[0],xplot.shape[0]), pred_mean, marker = '.', linestyle = '-', color ='b', markersize=10, label='GP Prediction')
        
    #     #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], 'r*', markersize=10, label='Actual Data')
    #     #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], marker = '*',linestyle = '-', color = 'r', markersize=10, label='Actual Data')
        
    #     axes[i].fill_between(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), 
    #                         (pred_mean - 1.96 * pred_std).flatten(), 
    #                         (pred_mean + 1.96 * pred_std).flatten(), 
    #                         color='orange', alpha=0.2, label='95% Confidence Interval')
    #     axes[i].set_title(f'{disturbance_d[j]} axis Disturbance vs {dim_d[i]}')
    #     axes[i].set_xlabel(dim_d[i])
    #     axes[i].set_ylabel('Disturbance (m/s^2)')
    #     axes[i].legend()
    
        # sorted_indices = jnp.argsort(xtest[:, i][plot_indices])
        
        # x_sorted = xtest[:, i][plot_indices][sorted_indices]
        # pred_mean_sorted = pred_mean[plot_indices][sorted_indices]
        # pred_std_sorted = pred_std[plot_indices][sorted_indices]
        
        #axes[i].plot(x_sorted, pred_mean_sorted, 'b.', markersize=10, label='GP Prediction')
    axes[j].plot(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), factor*actual_output, 'r*', markersize=10, label='Actual Data')
    axes[j].plot(np.linspace(0,xplot.shape[0],xplot.shape[0]), factor*pred_mean, marker = '.', linestyle = '-', color ='b', markersize=10, label='GP Prediction')
    
    #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], 'r*', markersize=10, label='Actual Data')
    #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], marker = '*',linestyle = '-', color = 'r', markersize=10, label='Actual Data')
    
    axes[j].fill_between(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), 
                        factor*(pred_mean - 1.96 * pred_std).flatten(), 
                        factor*(pred_mean + 1.96 * pred_std).flatten(), 
                        color='orange', alpha=0.2, label='95% Confidence Interval')
    axes[j].set_title(f'{disturbance_d[j]} axis Disturbance vs time index')
    axes[j].set_xlabel('time index')
    axes[j].set_ylabel('Disturbance (m/s^2)')
    axes[j].legend()

fig.tight_layout()
#fig.text(f"Mean Sqaure Error = {error_x+error_y+error_z}")
#plt.savefig(f"/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/GP_plots/disturbance_{disturbance_d[j]}.png")
plt.savefig("GP with normalized disturbance by factor of 3.png")
plt.show()


for j in range(3):
    fig = plt.figure(j)
    if j == 0:
        actual_output = wind_disturbance_curr = wind_disturbance_x
        pred_mean = pred_mean_x
        pred_std = pred_std_x
    elif j == 1:
        actual_output = wind_disturbance_curr = wind_disturbance_y
        pred_mean = pred_mean_y
        pred_std = pred_std_y
    elif j == 2:
        actual_output = wind_disturbance_curr = wind_disturbance_z
        pred_mean = pred_mean_z
        pred_std = pred_std_z
    
    plt.plot(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), factor* actual_output, 'r*', markersize=10, label='Actual Data')
    plt.plot(np.linspace(0,xplot.shape[0],xplot.shape[0]), factor*pred_mean, marker = '.', linestyle = '-', color ='b', markersize=10, label='GP Prediction')
    
    #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], 'r*', markersize=10, label='Actual Data')
    #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], marker = '*',linestyle = '-', color = 'r', markersize=10, label='Actual Data')
    
    plt.fill_between(np.linspace(0,actual_output.shape[0],actual_output.shape[0]), 
                        factor*(pred_mean - 1.96 * pred_std).flatten(), 
                        factor*(pred_mean + 1.96 * pred_std).flatten(), 
                        color='orange', alpha=0.2, label='95% Confidence Interval')
    
    plt.title(f'{disturbance_d[j]} axis Disturbance vs time index')
    plt.xlabel('time index')
    plt.ylabel('Disturbance (m/s^2)')
    plt.legend()
    plt.savefig(f'GP_norm3_{disturbance_d[j]}.png')
    plt.show()

# for j in range(3):
#     fig = plt.figure(3+j)

#     plt.set_title(f'{disturbance_d[j]} axis Disturbance vs time index')
#     plt.set_xlabel('time index')
#     plt.set_ylabel('Disturbance (m/s^2)')
#     plt.legend()
#     plt.show()


print("the means of disturbance are: ", jnp.mean(wind_disturbance,axis=0))
print("median of disturbance", jnp.median(wind_disturbance, axis=0))

print("predicted mean array is ", jnp.array([disturbance_x_mean, disturbance_y_mean, disturbance_z_mean]))
print("predicted median array is ", jnp.array([disturbance_x_median,disturbance_y_median, disturbance_z_median]))

########################################### Plotting ##########################################

    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust subplot grid as needed
    # axes = axes.flatten()
    # test_set_size = xtest.shape[0]
    # plot_size = plot_n = 100
    # plot_indices = jr.choice(key, test_set_size, (plot_n,) , replace=False)
    
    # for i in range(dim):
    #     sorted_indices = jnp.argsort(xtest[:, i][plot_indices])
        
    #     x_sorted = xtest[:, i][plot_indices][sorted_indices]
    #     pred_mean_sorted = pred_mean[plot_indices][sorted_indices]
    #     pred_std_sorted = pred_std[plot_indices][sorted_indices]

    #     #axes[i].plot(x_sorted, pred_mean_sorted, 'b.', markersize=10, label='GP Prediction')
    #     axes[i].plot(x_sorted, pred_mean_sorted, marker = '.', linestyle = '-', color ='b', markersize=10, label='GP Prediction')
    #     axes[i].plot(xtest[:, i][plot_indices], actual_output[plot_indices], 'r*', markersize=10, label='Actual Data')
    #     #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], 'r*', markersize=10, label='Actual Data')
    #     #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], marker = '*',linestyle = '-', color = 'r', markersize=10, label='Actual Data')
        
    #     axes[i].fill_between(x_sorted.flatten(), 
    #                         (pred_mean_sorted - 1.96 * pred_std_sorted).flatten(), 
    #                         (pred_mean_sorted + 1.96 * pred_std_sorted).flatten(), 
    #                         color='orange', alpha=0.2, label='95% Confidence Interval')
    #     axes[i].set_title(f'{disturbance_d[j]} axis Disturbance vs {dim_d[i]}')
    #     axes[i].set_xlabel(dim_d[i])
    #     axes[i].set_ylabel('Disturbance (m/s^2)')
    #     axes[i].legend()


    
   

    # fig.tight_layout()
    # plt.savefig(f"/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/GP_plots/disturbance_{disturbance_d[j]}.png")
    # plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(x, y, 'r.', markersize=10, label='Actual Data')
# plt.plot(xtest[:,0],pred_mean, 'b-', label='GP Prediction')
# plt.fill_between(xtest[:,0], pred_mean.flatten() - 1.96 * pred_std.flatten(),
#                  pred_mean.flatten() + 1.96 * pred_std.flatten(), color='blue', alpha=0.2, label='95% Confidence Interval')
# # plt.fill_between(xtest.flatten(), pred_mean.flatten() - 1.96 * pred_std.flatten(),
# #                  pred_mean.flatten() + 1.96 * pred_std.flatten(), color='blue', alpha=0.2, label='95% Confidence Interval')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.title('Gaussian Process Regression with GPJax')
# plt.legend()
# plt.show()

################################################# Plotting 3D Heatmap ##########################################
    # X = x
    # X_test = xtest
    # fig = plt.figure()
    # ax = fig.add_subplot(211, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', label='Data')
    # ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=pred_mean, cmap='magma', alpha=0.1, label='Predictions')

    # ax.set_zlim(-0.2,-0.6)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.title('Gaussian Process Regression on 3D Data')
    # plt.legend()
    # # plt.show()

    # ax = fig.add_subplot(212, projection='3d')
    # ax.set_zlim(-0.2,-0.6)
    # ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=pred_mean-actual_output, cmap='plasma', alpha=0.1, label='Trajectory')
    # plt.savefig(f"/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/GP_plots/disturbance_3d_heatmap_{disturbance_d[j]}.png")
    # plt.show()

# Perform the comparison and update
compare_and_update(file_path, error_x+error_y+error_z)