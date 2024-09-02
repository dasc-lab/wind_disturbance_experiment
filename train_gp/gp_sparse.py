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



key = jr.key(123)

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