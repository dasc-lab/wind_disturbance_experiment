from jax import config
config.update("jax_enable_x64", True)
import sys
import os
import pickle
import tensorflow_probability.substrates.jax.bijectors as tfb
import numpy as np
import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from gpjax.kernels import SumKernel, White, RBF, Matern32, RationalQuadratic, Periodic, ProductKernel
import matplotlib.pyplot as plt

training_path = 'datasets/training/'
wind_disturbance_training = np.load(training_path + 'disturbance.npy')
training_input = np.load(training_path+'input.npy')
wind_disturbance_x, wind_disturbance_y, wind_disturbance_z = wind_disturbance_training[:,0], wind_disturbance_training[:,1], wind_disturbance_training[:,2]
n = training_input.shape[0]
key = jr.key(123)

for j in range(3):
    if j == 0:
        wind_disturbance_curr = wind_disturbance_x
    elif j == 1:
        wind_disturbance_curr = wind_disturbance_y
    elif j == 2:
        wind_disturbance_curr = wind_disturbance_z
    
    x = training_input
    y = wind_disturbance_training
    

    ########################################## GP ##########################################


    D = gpx.Dataset(X=x, y=y)
    noise_level = 0.1
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

    likelihood = gpx.likelihoods.Gaussian(num_datapoints = n)

   
    posterior = prior * likelihood


    # Sparse part here ###############################################################
    # n_inducing = 50
    # z = jnp.linspace(-3.0, 3.0, n_inducing).reshape(-1, 1)
    # z = x[::10]
    z = x[::140]

    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=z
    )
    elbo = gpx.objectives.CollapsedELBO(negative=True)
    elbo = jit(elbo)

    opt_posterior, history = gpx.fit(
        model=q,
        objective=elbo,
        train_data=D,
        optim=ox.adamw(learning_rate=1e-2),
        num_iters=2000,
        key=key,
    )
    gp_model_file_path = 'models/'
    if j ==0:
        gp_model_x = opt_posterior
        #gp_model_file_path = home_path + 'gpmodels/gp_model_x_norm3.pkl'
        gp_model_file_path = gp_model_file_path + 'sparsegp_model_x_norm5_clipped_moredata.pkl'
    if j ==1:
        gp_model_y = opt_posterior
        #gp_model_file_path = home_path + 'gpmodels/gp_model_y_norm3.pkl'
        gp_model_file_path = gp_model_file_path + 'sparsegp_model_y_norm5_clipped_moredata.pkl'
    if j == 2:
        gp_model_z = opt_posterior
        gp_model_file_path = gp_model_file_path + 'sparsegp_model_z_norm5_clipped_moredata.pkl'
    with open(gp_model_file_path, 'wb') as file:
        pickle.dump(opt_posterior, file)