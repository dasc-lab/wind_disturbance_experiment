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

