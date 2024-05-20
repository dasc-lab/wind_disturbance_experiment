from jax import config

config.update("jax_enable_x64", True)
import sys
import os
plotter_path = os.path.join('/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/')
sys.path.append(plotter_path)
from plotter.plot_trajectory_ref_circle import cutoff, threshold

print(cutoff,threshold)
import tensorflow_probability.substrates.jax.bijectors as tfb
from simple_pytree import static_field
import numpy as np
import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from gpjax.kernels import SumKernel, White, RBF, Matern32, RationalQuadratic, Periodic
import matplotlib.pyplot as plt
from gpjax.kernels import AbstractKernel 
from dataclasses import dataclass
from typing import Any
from jaxtyping import Float, Array, install_import_hook

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base.param import param_field


def angular_distance(x, y, c):
    return jnp.abs((x - y + c) % (c * 2) - c)


bij = tfb.SoftClip(low=jnp.array(4.0, dtype=jnp.float64))


@dataclass
class Polar(gpx.kernels.AbstractKernel):
    period: float = static_field(2 * jnp.pi)
    tau: float = param_field(jnp.array([5.0]), bijector=bij)

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        c = self.period / 2.0
        t = angular_distance(x, y, c)
        K = (1 + self.tau * t / c) * jnp.clip(1 - t / c, 0, jnp.inf) ** self.tau
        return K.squeeze()
################################### Declare Parameters ##########################################

key = jr.key(123)

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


dim = 6 ## 6 input dims x,y,z,vx,vy,vz
################################### Data Prep ##########################################
wind_disturbance = jnp.load("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/training/disturbance.npy")
input = jnp.load("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/training/input_to_gp.npy")
assert wind_disturbance.shape[0] == input.shape[0]
#cutoff = wind_disturbance.shape[0]-3000
#threshold = 300
print("imported cutoff, threshold = ", cutoff, threshold)
set_size = cutoff - threshold
wind_disturbance = wind_disturbance[threshold:cutoff,:]
input = input[threshold:cutoff,:]
#print(wind_disturbance.shape)
assert wind_disturbance.shape[0] == input.shape[0]
wind_disturbance_x = jnp.array(wind_disturbance[:,0]).reshape(-1,1)
wind_disturbance_y = jnp.array(wind_disturbance[:,1]).reshape(-1,1)
wind_disturbance_z = jnp.array(wind_disturbance[:,2]).reshape(-1,1)
print("disturbance shape", wind_disturbance_x.shape)
assert wind_disturbance_x.shape == wind_disturbance_y.shape == wind_disturbance_z.shape == (set_size,1)

training_size = n = 800
indices = jr.choice(key, wind_disturbance_x.size, (training_size,) , replace=False)





for j in range(3):
    if j == 0:
        wind_disturbance_curr = wind_disturbance_x
    elif j == 1:
        wind_disturbance_curr = wind_disturbance_y
    elif j == 2:
        wind_disturbance_curr = wind_disturbance_z
    else:
        print("shouldn't reach this statement")
    x = input[indices]
    y = wind_disturbance_curr[indices]

    ########################################## GP ##########################################
    D = gpx.Dataset(X=x, y=y)
    noise_level = 0.5
    # Construct the prior
    meanf = gpx.mean_functions.Zero()
    white_kernel = White(variance=noise_level)
    kernel = SumKernel(kernels=[RBF(), Matern32(), white_kernel])
    kernel = SumKernel(kernels=[RBF(), Matern32()])
    rbf_kernel = RBF()
    rational_quadratic_kernel = RationalQuadratic()
    periodic_kernel = Periodic()
    combined_kernel = rbf_kernel * rational_quadratic_kernel * periodic_kernel
    kernel = combined_kernel
    #kernel = Polar()
    #kernel = RBF()
    prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)

    # Define a likelihood
    likelihood = gpx.likelihoods.Gaussian(num_datapoints = n)

    # Construct the posterior
    posterior = prior * likelihood

    # Define an optimiser
    optimiser = ox.adam(learning_rate=1e-2)

    # Define the marginal log-likelihood
    negative_mll = jit(gpx.objectives.ConjugateMLL(negative=True))

    # Obtain Type 2 MLEs of the hyperparameters
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=D,
        optim=optimiser,
        num_iters=500,
        safe=True,
        key=key,
    )

    print("after training, predicting")
    # Infer the predictive posterior distribution
    xtest = input
    latent_dist = opt_posterior(xtest, D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    # Obtain the predictive mean and standard deviation
    pred_mean = predictive_dist.mean()
    pred_std = predictive_dist.stddev()



########################################### Plotting ##########################################

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust subplot grid as needed
    axes = axes.flatten()
    plot_size = plot_n = 100
    plot_indices = jr.choice(key, set_size, (plot_n,) , replace=False)
    for i in range(dim):

        sorted_indices = jnp.argsort(xtest[:, i][plot_indices])
        x_sorted = xtest[:, i][plot_indices][sorted_indices]
        pred_mean_sorted = pred_mean[plot_indices][sorted_indices]
        pred_std_sorted = pred_std[plot_indices][sorted_indices]

        #axes[i].plot(x_sorted, pred_mean_sorted, 'b.', markersize=10, label='GP Prediction')
        axes[i].plot(x_sorted, pred_mean_sorted, marker = '.', linestyle = '-', color ='b', markersize=10, label='GP Prediction')
        axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], 'r*', markersize=10, label='Actual Data')
        #axes[i].plot(input[:, i][plot_indices], wind_disturbance_curr[plot_indices], marker = '*',linestyle = '-', color = 'r', markersize=10, label='Actual Data')
        
        axes[i].fill_between(x_sorted.flatten(), 
                            (pred_mean_sorted - 1.96 * pred_std_sorted).flatten(), 
                            (pred_mean_sorted + 1.96 * pred_std_sorted).flatten(), 
                            color='orange', alpha=0.2, label='95% Confidence Interval')
        axes[i].set_title(f'{disturbance_d[j]} axis Disturbance vs {dim_d[i]}')
        axes[i].set_xlabel(dim_d[i])
        axes[i].set_ylabel('Disturbance (m/s^2)')
        axes[i].legend()


        # sorted_indices = jnp.argsort(xtest[:, i])
        # x_sorted = xtest[:, i][sorted_indices]
        # pred_mean_sorted = pred_mean[sorted_indices]
        # pred_std_sorted = pred_std[sorted_indices]
        # x_plot = xtest[:,i][plot_indices]
        # pred_mean_plot = pred_mean[plot_indices]
        # assert x_plot.shape == input[:, i][plot_indices].shape
        # axes[i].plot(x_plot, pred_mean_plot, 'b.', markersize=10, label='GP Prediction')
        # axes[i].plot(input[:, i][plot_indices], wind_disturbance_x[plot_indices], 'r.', markersize=10, label='Actual Data')
        # axes[i].fill_between(x_sorted.flatten(), 
        #                      (pred_mean_sorted - 1.96 * pred_std_sorted).flatten(), 
        #                      (pred_mean_sorted + 1.96 * pred_std_sorted).flatten(), 
        #                      color='orange', alpha=0.2, label='95% Confidence Interval')
        # axes[i].set_title(f'Disturbance vs {dim_d[i]}')
        # axes[i].set_xlabel(dim_d[i])
        # axes[i].set_ylabel('Disturbance (m/s^2)')
        # axes[i].legend()
   

    fig.tight_layout()
    plt.savefig(f"/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/GP_plots/disturbance_{disturbance_d[j]}.png")
    plt.show()

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
    X = x
    X_test = xtest
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', label='Data')
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=pred_mean, cmap='magma', alpha=0.1, label='Predictions')

    ax.set_zlim(-0.2,-0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Gaussian Process Regression on 3D Data')
    plt.legend()
    # plt.show()

    ax = fig.add_subplot(212, projection='3d')
    ax.set_zlim(-0.2,-0.6)
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=wind_disturbance_curr, cmap='plasma', alpha=0.1, label='Trajectory')
    plt.savefig(f"/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/GP_plots/disturbance_3d_heatmap_{disturbance_d[j]}.png")
    plt.show()
    