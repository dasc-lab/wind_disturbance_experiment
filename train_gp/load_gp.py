from jax import config
config.update("jax_enable_x64", True)
import pickle
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox
import matplotlib.pyplot as plt
import numpy as np
import os



gp_models_path = 'gp_models/'
training_path = 'datasets/training/'
test_path = 'datasets/testing/'

training_disturbance_path = 'datasets/training/'
training_disturbance_x_path = training_disturbance_path + 'training_disturbance_x.npy'
training_disturbance_y_path = training_disturbance_path + 'training_disturbance_y.npy'
training_disturbance_z_path = training_disturbance_path + 'training_disturbance_z.npy'

training_disturbance_x = np.load(training_disturbance_x_path)
training_disturbance_y = np.load(training_disturbance_y_path)
training_disturbance_z = np.load(training_disturbance_z_path)

gp_models_path = 'gp_models/'
counter = 0
for gp_models_path in os.listdir(gp_models_path):
    if counter == 0:
        with open(training_disturbance_x_path, 'rb') as f:
            training_disturbance_x = pickle.load(f)
    elif counter == 1:
        with open(training_disturbance_y_path, 'rb') as f:
            training_disturbance_y = pickle.load(f)
    elif counter == 2:
        with open(training_disturbance_z_path, 'rb') as f:
            training_disturbance_z = pickle.load(f) 
    counter = counter + 1
test_disturbance_path = 'datasets/testing'
test_disturbance_path_x = test_disturbance_path + 'test'