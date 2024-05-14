from jax import config

config.update("jax_enable_x64", True)
import numpy as np
import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from gpjax.kernels import SumKernel, White, RBF
import matplotlib.pyplot as plt
key = jr.key(123)

wind_disturbance = jnp.load("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/disturbance.npy")
input = jnp.load("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/input_to_gp.npy")
cutoff = 4000
print("wind disturbance shape", wind_disturbance.shape)
print("input shape", input.shape)
wind_disturbance = wind_disturbance[:cutoff,:]
input = input[:cutoff,:]
print("wind disturbance shape", wind_disturbance.shape)
print("input shape", input.shape)
n = 200
wind_disturbance_x = jnp.array(wind_disturbance[:,0]).reshape(-1,1)
indices = np.random.choice(wind_disturbance_x.size, n, replace=False)
x = input[indices]
y = wind_disturbance_x[indices]
print(y.shape)
print(x.shape)
# n = 50
# x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,1)).sort()
# y = f(x) + 10*jr.normal(key, shape=(n,1))
# print(x.shape)
# print(y.shape)
# x = jnp.arange(wind_disturbance.size)
# y = wind_disturbance
D = gpx.Dataset(X=x, y=y)
noise_level = 0.5
# Construct the prior
meanf = gpx.mean_functions.Zero()
white_kernel = White(variance=noise_level)
kernel = SumKernel(kernels=[RBF(), white_kernel])
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

print("after training")
# Infer the predictive posterior distribution
xtest = input
latent_dist = opt_posterior(xtest, D)
predictive_dist = opt_posterior.likelihood(latent_dist)

# Obtain the predictive mean and standard deviation
pred_mean = predictive_dist.mean()
pred_std = predictive_dist.stddev()


fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust subplot grid as needed
axes = axes.flatten()
for i in range(6):
    axes[i].plot(xtest[:, i], pred_mean, 'b-',label='GP Prediction')
    axes[i].plot(x[:,i], y, 'r.', markersize=10, label='Actual Data')
    axes[i].fill_between(xtest[:, i], 
                         (pred_mean - 2 * pred_std).flatten(), 
                         (pred_mean + 2 * pred_std).flatten(), 
                         color='gray', alpha=0.2)
    
    axes[i].set_title(f'Dimension {i+1}')
    axes[i].set_xlabel(f'Input {i+1}')
    axes[i].set_ylabel('Predicted Output')

fig.tight_layout()
plt.savefig("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/GP_plots.png")
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