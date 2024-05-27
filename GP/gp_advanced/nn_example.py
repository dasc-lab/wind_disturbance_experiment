# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)

from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
from scipy.signal import sawtooth
from gpjax.base import static_field

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base import param_field
    import gpjax.kernels as jk
    from gpjax.kernels import DenseKernelComputation
    from gpjax.kernels.base import AbstractKernel
    from gpjax.kernels.computations import AbstractKernelComputation

key = jr.key(123)
# plt.style.use(
#     "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
# )
# cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
cols = [(255,0,0),(0,0,255),(0,0,0)]
n = 500
noise = 0.2

key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n,)).reshape(-1, 1)
f = lambda x: jnp.asarray(sawtooth(2 * jnp.pi * x))
signal = f(x)
y = signal + jr.normal(subkey, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-2.0, 2.0, 500).reshape(-1, 1)
ytest = f(xtest)

fig, ax = plt.subplots()
ax.plot(x, y, "o", label="Training data", alpha=0.5)
ax.plot(xtest, ytest, label="True function")
ax.legend(loc="best")
plt.show()
@dataclass
class DeepKernelFunction(AbstractKernel):
    base_kernel: AbstractKernel = None
    network: nn.Module = static_field(None)
    dummy_x: jax.Array = static_field(None)
    key: jax.Array = static_field(jr.key(123))
    nn_params: Any = field(init=False, repr=False)

    def __post_init__(self):
        if self.base_kernel is None:
            raise ValueError("base_kernel must be specified")
        if self.network is None:
            raise ValueError("network must be specified")
        self.nn_params = flax.core.unfreeze(self.network.init(key, self.dummy_x))

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, "1"]:
        state = self.network.init(self.key, x)
        xt = self.network.apply(state, x)
        yt = self.network.apply(state, y)
        return self.base_kernel(xt, yt)
feature_space_dim = 3


class Network(nn.Module):
    """A simple MLP."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=feature_space_dim)(x)
        return x


forward_linear = Network()
base_kernel = gpx.kernels.Matern52(
    active_dims=list(range(feature_space_dim)),
    lengthscale=jnp.ones((feature_space_dim,)),
)
kernel = DeepKernelFunction(
    network=forward_linear, base_kernel=base_kernel, key=key, dummy_x=x
)
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
posterior = prior * likelihood
schedule = ox.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=0.01,
    warmup_steps=75,
    decay_steps=700,
    end_value=0.0,
)

optimiser = ox.chain(
    ox.clip(1.0),
    ox.adamw(learning_rate=schedule),
)

opt_posterior, history = gpx.fit(
    model=posterior,
    objective=jax.jit(gpx.objectives.ConjugateMLL(negative=True)),
    train_data=D,
    optim=optimiser,
    num_iters=800,
    key=key,
)
latent_dist = opt_posterior(xtest, train_data=D)
predictive_dist = opt_posterior.likelihood(latent_dist)

predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

fig, ax = plt.subplots()
ax.plot(x, y, "o", label="Observations", color='red')
ax.plot(xtest, predictive_mean, label="Predictive mean", color='blue')
ax.fill_between(
    xtest.squeeze(),
    predictive_mean - 2 * predictive_std,
    predictive_mean + 2 * predictive_std,
    alpha=0.2,
    color='orange',
    label="Two sigma",
)
ax.plot(
    xtest,
    predictive_mean - 2 * predictive_std,
    color='red',
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest,
    predictive_mean + 2 * predictive_std,
    color='red',
    linestyle="--",
    linewidth=1,
)
ax.legend()
plt.show()