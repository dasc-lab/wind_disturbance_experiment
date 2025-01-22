import time
import jax
import numpy as np
from jax import random, grad, jit, lax, value_and_grad, jacrev, jacfwd
import optax
import jaxopt
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})
# gpjax version: '0.8.2'
import gpjax as gpx
from test_jax_utils import *
from test_gp_utils_angles import *
from test_policy_with_obstacle import policy
plt.rcParams.update({'font.size': 14})

num_particles = 1000

name = "rebuttal_"

# dynamics_type = 'ideal'
# dynamics_type = 'noisy'
dynamics_type = 'gp'
obs_center = jnp.array([-0.4,0.2,-0.5]).reshape(-1,1)
# optimizer = 'scipy'
optimizer = 'custom_gd'

#home_path = '/home/dasc/albus/wind_disturbance_experiment/GP/gp_advanced/'
# home_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/'
home_path = '/home/hardik/Desktop/Research/wind_disturbance_experiment/GP/gp_advanced/'
# home_path = '/home/wind_disturbance_experiment/GP/gp_advanced/'

# trajectory_path = home_path + 'circle_figure8_fullset/'
trajectory_path = home_path + 'gp_final/'
model_path = trajectory_path + 'models/'
disturbance_path = trajectory_path + 'disturbance_new.npy'
input_path = trajectory_path + 'input_new.npy'

home_path = '/home/hardik/Desktop/Research/wind_disturbance_experiment/train_gp_withacmd/'
model_path = home_path + 'gp_models/'
disturbance_path = home_path + 'datasets/all_data/disturbance.npy'
input_path = home_path + 'datasets/all_data/input.npy'

key = random.PRNGKey(2)
horizon = 60 #100 #60 #50 #300 #200
simT = 300
predict_dt = 0.05 #0.1 #0.05 #0.01
optimize_dt = 0.05



# custom optimizer
iter_adam_custom = 3 #300
custom_gd_lr_rate = 0.2 #0.05
grad_clip = 2.0
violation_factor = 2#20000
custom_gd_lr_rate_reward = custom_gd_lr_rate / 1
# 1.0: 1/4,                                       # tanh inside parameter vs
custom_gd_lr_rate_violation = custom_gd_lr_rate * 2
# 1.0: 2.0, 

# params_init = jnp.array([7.0, 4.0, 4.0, 0.1])
params_init = jnp.array([7.0, 4.0, 2.0, 1.0])
W1 = 0 #0.01
W2 = 0 #0.01

# scipy optimizer
iter_scipy=1 #4000

def initialize_sigma_points(X):
    '''
    Returns Equally weighted Sigma Particles
    '''
    n = X.shape[0]
    num_points = 2*n + 1
    X = X.reshape(-1,1)
    sigma_points = jnp.repeat( X, num_points, axis=1 )
    weights = jnp.ones((1,num_points)) * 1.0/( num_points )
    return sigma_points, weights

def reward_func(states, weights, pos_ref, vel_ref):
    '''
    calculates mean squared error
    inputs: states and the weights of sigma points
    returns: calculated reward
    '''
    ex = states[0:3] - pos_ref
    ev = states[3:6] - vel_ref

    ex_ev_mean = get_mean(jnp.append(ex, ev, axis=0), weights )

    pos_factor = 1.0
    vel_factor = 0.1
    reward = pos_factor * jnp.sum(ex_ev_mean[0:3] ** 2) + vel_factor * jnp.sum(ex_ev_mean[3:6] ** 2)
    return reward

@jit
def constraint_violation(states, weights, circle_center, circle_radius):

    dists = jnp.linalg.norm(states[0:2]-circle_center[0:2], axis=0).reshape((1,13)) - circle_radius
    mean_dist, cov_dist = get_mean_cov( dists, weights )
    risk_dist = mean_dist - 2.96 * jnp.sqrt(cov_dist)
    risk_dist = jnp.clip( risk_dist, None, 0.0 )
    # jax.debug.print("{x}", x=risk_dist)
    # slack = jnp.min( jnp.array([ 0.0, risk_dist[0,0]]) )  # 0 if safe, negative is unsafe
    # return -slack
    return risk_dist[0,0]

@jit
def constraint_violation_predict(states, weights, circle_center, circle_radius):

    dist = jnp.linalg.norm(states[0:2,0]-circle_center[0:2,0], axis=0)-circle_radius
    # slack = jnp.min( jnp.array([ 0.0, dist]) )  # 0 if safe, negative is unsafe
    # return -slack 
    return dist



@jit
def predict_state_ideal(state, policy_params, key, h):
    t = h * predict_dt
    control_input, pos_ref, vel_ref = policy( t, state, policy_params )         # mean_position = get_mean( states, weights )
    next_state, _ = get_next_states_ideal( state, control_input, predict_dt )
    state_ref = jnp.append(pos_ref, vel_ref, axis=0)
    disturbance_mean, disturbance_cov = jnp.zeros((3,1)), jnp.zeros((3,1))
    return next_state, state_ref, control_input, disturbance_mean, disturbance_cov

@jit
def predict_state_noisy(state, policy_params, key, h):

    t = h * predict_dt
    control_input, pos_ref, vel_ref = policy( t, state, policy_params )         # mean_position = get_mean( states, weights )
    next_state, next_state_cov, disturbance_mean, disturbance_cov = get_next_states_noisy_predict( state, control_input, predict_dt )
    key, subkey = jax.random.split(key)
    next_state = next_state + jax.random.normal( subkey, shape=(6,1) ) * jnp.sqrt( next_state_cov )
    # next_states = next_states + jax.random.normal( subkey, shape=(6,13) ) * jnp.sqrt( next_states_cov )
    state_ref = jnp.append(pos_ref, vel_ref, axis=0)
    return next_state, state_ref, control_input, disturbance_mean, disturbance_cov

# @jit
def setup_predict_state_gp(file_path1, file_path2, file_path3, x, gp_train_y): #, dt):

    gp0 = initialize_gp_prediction( file_path1 ) #, gp_train_x, gp_train_y[:,0].reshape(-1,1) )
    gp1 = initialize_gp_prediction( file_path2 ) #, gp_train_x, gp_train_y[:,1].reshape(-1,1) )
    gp2 = initialize_gp_prediction( file_path3 ) #, gp_train_x, gp_train_y[:,2].reshape(-1,1) )

    x = gp_train_x
    y = gp_train_y

    ###### precomputes all necessary inverses to save time ######
    D0 = gpx.Dataset(X=x, y=y[0].reshape(-1,1))
    D1 = gpx.Dataset(X=x, y=y[1].reshape(-1,1))
    D2 = gpx.Dataset(X=x, y=y[2].reshape(-1,1))

    L0, L0_inv, Lz0, Lz_inv0, Kzz_inv_Kzx_diff0 = gp0.compute_sigma_inv(train_data=D0)
    L1, L1_inv, Lz1, Lz_inv1, Kzz_inv_Kzx_diff1 = gp1.compute_sigma_inv(train_data=D1)
    L2, L2_inv, Lz2, Lz_inv2, Kzz_inv_Kzx_diff2 = gp2.compute_sigma_inv(train_data=D2)

    @jit
    def predict_state(state, policy_params, key, h):
        
        t = h * predict_dt
        control_input, pos_ref, vel_ref = policy( t, state, policy_params )         # mean_position = get_mean( states, weights )
        next_state, next_state_cov, disturbance_mean, disturbance_cov = get_next_states_with_sparse_gp_sigma_inv_predict( state, control_input, predict_dt, [gp0, gp1, gp2], [L0, L1, L2], [L0_inv, L1_inv, L2_inv], [Lz0, Lz1, Lz2], [Lz_inv0, Lz_inv1, Lz_inv2], [Kzz_inv_Kzx_diff0, Kzz_inv_Kzx_diff1, Kzz_inv_Kzx_diff2])
        key, subkey = jax.random.split(key)
        next_state = next_state + jax.random.normal( subkey, shape=(6,1) ) * jnp.sqrt( next_state_cov )
        state_ref = jnp.append(pos_ref, vel_ref, axis=0)
        return next_state, state_ref, control_input, disturbance_mean, disturbance_cov
    
    return predict_state


def setup_future_reward_func(file_path1, file_path2, file_path3, dynamics_type='ideal'):

    gp0 = initialize_gp_prediction( file_path1 ) #, gp_train_x, gp_train_y[:,0].reshape(-1,1) )
    gp1 = initialize_gp_prediction( file_path2 ) #, gp_train_x, gp_train_y[:,1].reshape(-1,1) )
    gp2 = initialize_gp_prediction( file_path3 ) #, gp_train_x, gp_train_y[:,2].reshape(-1,1) )

    x = gp_train_x
    y = gp_train_y

    ###### precomputes all necessary inverses to save time ######
    D0 = gpx.Dataset(X=x, y=y[0].reshape(-1,1))
    D1 = gpx.Dataset(X=x, y=y[1].reshape(-1,1))
    D2 = gpx.Dataset(X=x, y=y[2].reshape(-1,1))
    L0, L0_inv, Lz0, Lz_inv0, Kzz_inv_Kzx_diff0 = gp0.compute_sigma_inv(train_data=D0)
    L1, L1_inv, Lz1, Lz_inv1, Kzz_inv_Kzx_diff1 = gp1.compute_sigma_inv(train_data=D1)
    L2, L2_inv, Lz2, Lz_inv2, Kzz_inv_Kzx_diff2 = gp2.compute_sigma_inv(train_data=D2)

    n = 6

    @jit
    def compute_reward(X, policy_params, init_time):
        '''
        Performs Gradient Descent
        '''
        sigma_states = jnp.zeros((n*(2*n+1),horizon+1))
        sigma_weights = jnp.zeros((2*n+1,horizon))
        mc_particles = jnp.zeros((8*num_particles, horizon+1))

        # obs_center = jnp.array([-0.3,0.1,-0.5]).reshape(-1,1)
        dist = constraint_violation_predict(X, jnp.ones((1,1)), obs_center, 0.4)
        # jax.debug.print("dist inside: {x}", x=dist)
        states, weights = initialize_sigma_points(X)

        # Store states
        sigma_states = sigma_states.at[:,0].set(states.reshape(-1,1, order='F')[:,0])
        sigma_weights = sigma_weights.at[:,0].set(weights.reshape(-1,1, order='F')[:,0])
        # mc_particles = mc_particles.at[:,0].set( jnp.tile(X, num_particles).reshape(-1,1, order='F')[:,0] )


        kx = policy_params[0]
        kv = policy_params[1]
        w1 = W1 #0.05#0.5
        w2 = W2 #0.05 #0.1
        # reward = 0 + w1 * (kx-7)**2 + w2 * (kv-4)**2
        reward = 0 + w1 * (kx)**2 + w2 * (kv)**2
        constraint = 0
        # reward = 0
        def body(h, inputs):
            '''
            Performs UT-EC with 6 states
            '''
            t = init_time + h * optimize_dt
            # jax.debug.print("time: {x}", x=t)
            reward, states, weights, constraint, sigma_states, sigma_weights = inputs
            control_inputs, pos_ref, vel_ref = policy( t, states, policy_params )         # mean_position = get_mean( states, weights )
            # control_inputs_mc, pos_ref_mc, vel_ref_mc = policy( t, mc_particles[:,h].reshape((  ), order='F'), policy_params )
            next_states_mean, next_states_cov = get_next_states_with_sparse_gp_sigma_inv( states, control_inputs, optimize_dt, [gp0, gp1, gp2], [L0, L1, L2], [L0_inv, L1_inv, L2_inv],  [Lz0, Lz1, Lz2], [Lz_inv0, Lz_inv1, Lz_inv2], [Kzz_inv_Kzx_diff0, Kzz_inv_Kzx_diff1, Kzz_inv_Kzx_diff2], [D0, D1, D2])
            
            next_states_expanded, next_weights_expanded = sigma_point_expand_with_mean_cov( next_states_mean, next_states_cov, weights)
            next_states, next_weights = sigma_point_compress( next_states_expanded, next_weights_expanded )

            sigma_states = sigma_states.at[:,h+1].set( next_states.reshape(-1,1, order='F')[:,0] )
            sigma_weights = sigma_weights.at[:,h+1].set( next_weights.reshape(-1,1, order='F')[:,0] )
            # next_mc_particles = 

            control_inputs, pos_ref, vel_ref = policy( t, states, policy_params ) 
            states = next_states
            weights = next_weights
            reward = reward + reward_func( states, weights, pos_ref, vel_ref ) # reward is loss
            circle_radius = 0.4
            constraint = constraint + constraint_violation( states, weights, obs_center, circle_radius )
            return reward, states, weights, constraint, sigma_states, sigma_weights
        reward, _, _, constraint, sigma_states, sigma_weights =  lax.fori_loop( 0, horizon, body, (reward, states, weights, constraint, sigma_states, sigma_weights) )
        # jax.debug.print("{x}", x=risk_dist)
        # return jnp.array([reward, constraint]), sigma_states, sigma_weights
        return sigma_states, sigma_weights
    return compute_reward


def setup_future_reward_func_mc(file_path1, file_path2, file_path3, dynamics_type='ideal'):

    gp0 = initialize_gp_prediction( file_path1 ) #, gp_train_x, gp_train_y[:,0].reshape(-1,1) )
    gp1 = initialize_gp_prediction( file_path2 ) #, gp_train_x, gp_train_y[:,1].reshape(-1,1) )
    gp2 = initialize_gp_prediction( file_path3 ) #, gp_train_x, gp_train_y[:,2].reshape(-1,1) )

    x = gp_train_x
    y = gp_train_y

    ###### precomputes all necessary inverses to save time ######
    D0 = gpx.Dataset(X=x, y=y[0].reshape(-1,1))
    D1 = gpx.Dataset(X=x, y=y[1].reshape(-1,1))
    D2 = gpx.Dataset(X=x, y=y[2].reshape(-1,1))
    L0, L0_inv, Lz0, Lz_inv0, Kzz_inv_Kzx_diff0 = gp0.compute_sigma_inv(train_data=D0)
    L1, L1_inv, Lz1, Lz_inv1, Kzz_inv_Kzx_diff1 = gp1.compute_sigma_inv(train_data=D1)
    L2, L2_inv, Lz2, Lz_inv2, Kzz_inv_Kzx_diff2 = gp2.compute_sigma_inv(train_data=D2)

    n = 6

    @jit
    def compute_reward(X, policy_params, init_time):
        '''
        Performs Gradient Descent
        '''
        mc_particles = jnp.zeros((n*num_particles, horizon+1))

        # Store states
        mc_particles = mc_particles.at[:,0].set( jnp.tile(X, num_particles).reshape(-1,1, order='F')[:,0] )

        key = jax.random.PRNGKey(0)

        kx = policy_params[0]
        kv = policy_params[1]
        w1 = W1 #0.05#0.5
        w2 = W2 #0.05 #0.1
        # reward = 0 + w1 * (kx-7)**2 + w2 * (kv-4)**2
        reward = 0 + w1 * (kx)**2 + w2 * (kv)**2
        constraint = 0
        # reward = 0
        def body(h, inputs):
            '''
            Performs UT-EC with 6 states
            '''
            t = init_time + h * optimize_dt
            # jax.debug.print("time: {x}", x=t)
            mc_particles, key = inputs
            states = mc_particles[:,h].reshape((n , num_particles ), order='F')
            control_inputs_mc, pos_ref_mc, vel_ref_mc = policy( t, states, policy_params )
            next_states, key = get_next_states_with_sparse_gp_sigma_inv_mc( states, control_inputs_mc, optimize_dt, [gp0, gp1, gp2], [L0, L1, L2], [L0_inv, L1_inv, L2_inv],  [Lz0, Lz1, Lz2], [Lz_inv0, Lz_inv1, Lz_inv2], [Kzz_inv_Kzx_diff0, Kzz_inv_Kzx_diff1, Kzz_inv_Kzx_diff2], [D0, D1, D2], key)
            
            # sigma_states = sigma_states.at[:,h+1].set( next_states.reshape(-1,1, order='F')[:,0] )
            # sigma_weights = sigma_weights.at[:,h+1].set( next_weights.reshape(-1,1, order='F')[:,0] )

            mc_particles = mc_particles.at[ :,h+1 ].set( next_states.reshape(-1,1, order='F')[:,0] )

            return mc_particles, key
        mc_particles, key=  lax.fori_loop( 0, horizon, body, (mc_particles, key) )
        # jax.debug.print("{x}", x=risk_dist)
        # return jnp.array([reward, constraint]), sigma_states, sigma_weights
        return mc_particles, key
    return compute_reward



print(model_path)
file_path1 = model_path + "sparsegp_model_x_norm5_clipped.pkl"
file_path2 = model_path + "sparsegp_model_y_norm5_clipped.pkl"
file_path3 = model_path + "sparsegp_model_z_norm5_clipped.pkl"


gp_train_x = jnp.load(input_path)
# gp_train_x = gp_train_x[::80]
gp_train_x = gp_train_x#[::140]
gp_train_y = jnp.load(disturbance_path)
gp_train_y = gp_train_y.T#[::140].T
t0 = time.time()
# import pdb
# pdb.set_trace()
# def generate_state_vector(key, n):
#     return jax.random.normal(key, (n, 1))
def generate_state_vector(key, n):
    state_vector = jax.random.normal(key, (n, 1))
    state_vector = state_vector.at[0,0].set(0.0)
    state_vector = state_vector.at[1,0].set(0.0)
    return state_vector

# Example usage:
key = jax.random.PRNGKey(0)  # Initialize the random key
n = 6  # Size of the state vector
state_vector = generate_state_vector(key, n)
state_vector = jnp.array([0.0,0,0,0,0,0]).reshape(-1,1)
print(state_vector)

def setup_predict_states(file_path1, file_path2, file_path3, dynamics_type='ideal'):
    if dynamics_type=='ideal':
        return predict_state_ideal
    elif dynamics_type=='noisy':
        return predict_state_noisy
    elif dynamics_type=='gp':
        predict_states_gp = setup_predict_state_gp(file_path1, file_path2, file_path3, gp_train_x, gp_train_y)
        return predict_states_gp

# first run
get_future_reward = setup_future_reward_func(file_path1, file_path2, file_path3, dynamics_type=dynamics_type)
get_future_reward_mc = setup_future_reward_func_mc(file_path1, file_path2, file_path3, dynamics_type=dynamics_type)
# predict_state = setup_predict_states(file_path1, file_path2, file_path3, dynamics_type=dynamics_type)

get_future_reward( state_vector, jnp.array([7.0, 4.0, 1.0, 0.1]), 0.0 )
get_future_reward_mc( state_vector, jnp.array([7.0, 4.0, 1.0, 0.1]), 0.0 )
key, subkey = jax.random.split(key)


# Unoptimized Parameters
fig, ax = plt.subplots()

t0 = time.time()
sigma_states, sigma_weights = get_future_reward( state_vector, jnp.array([7.0, 4.0, 1.0, 0.1]), 0.0 )
t1 = time.time()
print(f"time: {time.time()-t0}")
t0 = time.time()
sigma_states, sigma_weights = get_future_reward( state_vector, jnp.array([7.0, 4.0, 1.0, 0.1]), 0.0 )
t1 = time.time()
print(f"time: {time.time()-t0}")

t0 = time.time()
mc_particles, _ = get_future_reward_mc( state_vector, jnp.array([7.0, 4.0, 1.0, 0.1]), 0.0 )
t1 = time.time()
print(f"mc time: {time.time()-t0}")
t0 = time.time()
mc_particles, _ = get_future_reward_mc( state_vector, jnp.array([7.0, 4.0, 1.0, 0.1]), 0.0 )
t1 = time.time()
print(f"mc time: {time.time()-t0}")

for i in range(horizon):
    sigma_points = sigma_states[:,i].reshape((n, 2*n+1), order='F')
    # print(i)
    # print(sigma_points)
    ax.scatter(sigma_points[0,:], sigma_points[1,:], 50, label=f'{i}')
    mc_points = mc_particles[:,i].reshape((n, num_particles), order='F')
    ax.scatter(mc_points[0,0::10], mc_points[1,0::10], 10, alpha=0.3)




from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# obs_center = np.array([-0.4,0,-0.5]).reshape(-1,1)
# obs_center = np.array([-0.3,0.1,-0.5]).reshape(-1,1)
circ = plt.Circle((obs_center[0,0],obs_center[1,0]),0.4,linewidth = 1, edgecolor='k',facecolor='k', alpha=0.4)
ax.add_patch(circ)

ax.legend()
plt.show()


# fig.savefig(name+f"simulation_paths.png")
# fig.savefig(name+f"simulation_paths.eps")

# ax_acc[0].legend()
# ax_acc[1].legend()
# ax_acc[2].legend()
# 


import pdb
pdb.set_trace()






# fig_acc, ax_acc = plt.subplots(3)
# ax_acc[0].plot( control_inputs[0,:], 'g', label='Control acceleration' )
# ax_acc[1].plot( control_inputs[1,:], 'g' )
# ax_acc[2].plot( control_inputs[2,:], 'g' )
# ax_acc[0].plot( disturbance_means[0,:], 'r', label='Disturbance' )
# ax_acc[1].plot( disturbance_means[1,:], 'r' )
# ax_acc[2].plot( disturbance_means[2,:], 'r' )
# ax_acc[0].plot( control_inputs2[0,:], 'g--')
# ax_acc[1].plot( control_inputs2[1,:], 'g--' )
# ax_acc[2].plot( control_inputs2[2,:], 'g--' )
# ax_acc[0].plot( disturbance_means2[0,:], 'r--')
# ax_acc[1].plot( disturbance_means2[1,:], 'r--' )
# ax_acc[2].plot( disturbance_means2[2,:], 'r--' )
# ax_acc[0].set_ylabel('X')
# ax_acc[1].set_ylabel('Y')
# ax_acc[2].set_ylabel('Z')

# ax_acc[0].plot( control_inputs_optimized[0,:], 'b', label='Control acceleration optimized' )
# ax_acc[1].plot( control_inputs_optimized[1,:], 'b' )
# ax_acc[2].plot( control_inputs_optimized[2,:], 'b' )
# ax_acc[0].plot( disturbance_means_optimized[0,:], 'k', label='Disturbance optimized' )
# ax_acc[1].plot( disturbance_means_optimized[1,:], 'k' )
# ax_acc[2].plot( disturbance_means_optimized[2,:], 'k' )
# ax_acc[0].plot( control_inputs_optimized2[0,:], 'b--', label='Control acceleration optimized' )
# ax_acc[1].plot( control_inputs_optimized2[1,:], 'b--' )
# ax_acc[2].plot( control_inputs_optimized2[2,:], 'b--' )
# ax_acc[0].plot( disturbance_means_optimized2[0,:], 'k--', label='Disturbance optimized' )
# ax_acc[1].plot( disturbance_means_optimized2[1,:], 'k--' )
# ax_acc[2].plot( disturbance_means_optimized2[2,:], 'k--' )