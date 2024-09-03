from jax import config
config.update("jax_enable_x64", True)
import os
import tensorflow_probability.substrates.jax.bijectors as tfb
import numpy as np
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
current_file_path = os.path.abspath(__file__)
recorded_data_path = 'recorded_data/'
npy_path = 'datasets/'


all_data_path = npy_path + 'all_data/'
circle_path = recorded_data_path + 'circle_data/'

figure8_path = recorded_data_path +'figure8_data/'

bag_path = circle_path + '28_07_2024_take1_cir_traj_r0.4_w1.0_c0.00_h0.5_kxv19_04_9_30_tank_0_31_fanon_clipped_new'
input = np.load(all_data_path+'input.npy')
wind_disturbance = np.load(all_data_path+'disturbance.npy')
factor = 5
slice = 20
key = jr.key(123)

wind_disturbance = wind_disturbance[::slice]
input = input[::slice]

##################### Remove Outliers #####################
outlier_threshold = 5.0/factor
rows_to_remove = jnp.any(wind_disturbance > outlier_threshold, axis=1)
wind_disturbance = wind_disturbance[~rows_to_remove]
input = input[~rows_to_remove]
print(f"The remaining length is {wind_disturbance.shape}")

np.save(npy_path + "fullset_input.npy", input)
assert wind_disturbance.shape[0] == input.shape[0]


wind_disturbance_x = jnp.array(wind_disturbance[:,0]).reshape(-1,1)
wind_disturbance_y = jnp.array(wind_disturbance[:,1]).reshape(-1,1)
wind_disturbance_z = jnp.array(wind_disturbance[:,2]).reshape(-1,1)
# np.save(npy_path + "wind_disturbance_x.npy", wind_disturbance_x)
# np.save(npy_path + "wind_disturbance_y.npy", wind_disturbance_y)
# np.save(npy_path + "wind_disturbance_z.npy", wind_disturbance_z)
assert wind_disturbance_x.shape == wind_disturbance_y.shape == wind_disturbance_z.shape# == (set_size,1)




training_size = n = int(wind_disturbance.shape[0]*0.8)
training_indices = jr.choice(key, wind_disturbance_x.size, (training_size,) , replace=False)
mask = jnp.ones(wind_disturbance.shape[0], dtype=bool)
mask = mask.at[training_indices].set(False)
test_input = input[mask]
test_disturbance_x = wind_disturbance_x[mask]
test_disturbance_y = wind_disturbance_y[mask]
test_disturbance_z = wind_disturbance_z[mask]
testset_path = npy_path + 'testing/'
jnp.save(testset_path+'test_input.npy', test_input)
jnp.save(testset_path+'test_disturbance_x.npy', test_disturbance_x)
jnp.save(testset_path+'test_disturbance_y.npy', test_disturbance_y)
jnp.save(testset_path+'test_disturbance_z.npy', test_disturbance_z)

print(f"training on {training_size} datapoints")
training_input = input[training_indices]
training_path = npy_path + 'training/'
training_disturbance_x = wind_disturbance_x[training_indices]
training_disturbance_y = wind_disturbance_y[training_indices]
training_disturbance_z = wind_disturbance_z[training_indices]
np.save(training_path+"training_input.npy", training_input)
np.save(training_path+"training_disturbance_x.npy", training_disturbance_x)
np.save(training_path+"training_disturbance_y.npy", training_disturbance_y)
np.save(training_path+"training_disturbance_z.npy", training_disturbance_z)