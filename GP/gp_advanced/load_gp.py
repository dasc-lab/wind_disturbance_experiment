from jax import config

config.update("jax_enable_x64", True)

import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox

key = jr.key(123)

f = lambda x: 10 * jnp.sin(x)

n = 50
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,1)).sort()
y = f(x) + jr.normal(key, shape=(n,1))
D = gpx.Dataset(X=x, y=y)

# Construct the prior
meanf = gpx.mean_functions.Zero()
kernel = gpx.kernels.RBF()
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

# Infer the predictive posterior distribution
xtest = jnp.linspace(-3., 3., 100).reshape(-1, 1)
latent_dist = opt_posterior(xtest, D)
predictive_dist = opt_posterior.likelihood(latent_dist)

# Obtain the predictive mean and standard deviation
pred_mean = predictive_dist.mean()
pred_std = predictive_dist.stddev()
