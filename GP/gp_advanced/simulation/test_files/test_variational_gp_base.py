# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
# from docs.examples.utils import clean_legend

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

key = jr.key(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

n = 2500
noise = 0.5

key, subkey = jr.split(key)
x1 = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
x2 = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
x = jnp.append(x1, x2, axis=1)
f = lambda x: jnp.sin(2 * x[:,[0]]) + x[:,[1]] * jnp.cos(5 * x[:,[1]])
signal = f(x)
y = signal + jr.normal(subkey, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest1 = jnp.linspace(-3.1, 3.1, 500).reshape(-1, 1)
xtest2 = jnp.linspace(-3.1, 3.1, 500).reshape(-1, 1)
xtest = jnp.append(xtest1, xtest2, axis=1)
ytest = f(xtest)


fig, ax = plt.subplots()
ax.scatter(x[:,[0]], y, alpha=0.25, label="Observations", color=cols[0])
# ax.scatter(x[:,[1]], y, alpha=0.25, label="Observations", color=cols[2], alpha=0.2)
ax.plot(xtest, ytest, label="Latent function", linewidth=2, color=cols[1])
ax.legend(loc="best")
plt.show()

meanf = gpx.mean_functions.Constant()
kernel = gpx.kernels.RBF(active_dims=[0,1])
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
posterior = prior * likelihood
negative_mll = jit(gpx.objectives.ConjugateMLL(negative=True))

# full model
opt_posterior, history = gpx.fit(
    model=posterior,
    objective=negative_mll,
    train_data=D,
    optim=ox.adamw(learning_rate=1e-2),
    num_iters=5, #00,
    key=key,
)

import time

sigma0 = opt_posterior.compute_sigma_inv(train_data=D)

# latent_dist = opt_posterior.predict_with_sigma_inv(xtest, L, Lz, Kzz_inv_Kzx_diff)
# latent_dist = opt_posterior.predict(xtest, train_data=D)
# latent_dist = opt_posterior.predict_with_sigma_inv(xtest, L, Lz, Kzz_inv_Kzx_diff)
# latent_dist = opt_posterior.predict(xtest, train_data=D)
# latent_dist = opt_posterior.predict_with_sigma_inv(xtest, L, Lz, Kzz_inv_Kzx_diff)
# latent_dist = opt_posterior.predict(xtest, train_data=D)
# latent_dist = opt_posterior.predict_with_sigma_inv(xtest, L, Lz, Kzz_inv_Kzx_diff)
# latent_dist = opt_posterior.predict(xtest, train_data=D)

t0 = time.time()
latent_dist = opt_posterior.predict_with_sigma_inv(xtest, D, Sigma_inv=sigma0)
t1 = time.time()
print(f"time: {t1-t0}")

t0 = time.time()
latent_dist = opt_posterior.predict(xtest, train_data=D)
t1 = time.time()
print(f"time: {t1-t0}")


predictive_dist = opt_posterior.likelihood(latent_dist)
samples = latent_dist.sample(seed=key, sample_shape=(20,))
predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()


@jit
def pred1(xtest):
    latent_dist = opt_posterior.predict_with_sigma_inv(xtest, D, Sigma_inv=sigma0)
    predictive_dist = opt_posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()
    return predictive_std, predictive_mean


@jit
def pred2(xtest):
    latent_dist = opt_posterior.predict(xtest, train_data=D)
    predictive_dist = opt_posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()
    return predictive_std, predictive_mean

std1, mu1 = pred1(xtest)
std1, mu1 = pred1(xtest)
std2, mu2 = pred2(xtest)
std2, mu2 = pred2(xtest)

t0 = time.time()
std1, mu1 = pred1(xtest)
t1 = time.time()
print(f"jit time 1: {t1-t0}")

t0 = time.time()
std2, mu2 = pred2(xtest)
t1 = time.time()
print(f"jit time 2: {t1-t0}")

fig, ax = plt.subplots()

ax.plot(x, y, "x", label="Observations", color=cols[0], alpha=0.1)
ax.plot(
    xtest[:,0],
    ytest,
    label="Latent function",
    color=cols[1],
    linestyle="-",
    linewidth=1,
)
ax.plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])

ax.fill_between(
    xtest[:,0].squeeze(),
    predictive_mean - 2 * predictive_std,
    predictive_mean + 2 * predictive_std,
    alpha=0.2,
    color=cols[1],
    label="Two sigma",
)
ax.plot(
    xtest[:,0],
    predictive_mean - 2 * predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=0.5,
)
ax.plot(
    xtest[:,0],
    predictive_mean + 2 * predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=0.5,
)


ax.legend()
ax.set(xlabel=r"xxx", ylabel=r"f(x)f(x)f(x)")
plt.show()