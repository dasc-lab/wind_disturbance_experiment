import time
import jax
import jax.numpy as np
from jax import random, grad, jit, lax
import optax
import jaxopt
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})
# gpjax version: '0.8.2'

from test_jax_utils import *
from test_gp_utils import *
from test_policy import policy

#home_path = '/home/dasc/albus/wind_disturbance_experiment/GP/gp_advanced/'
# home_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/'
home_path = '/home/hardik/Desktop/Research/wind_disturbance_experiment/GP/gp_advanced/'

trajectory_path = home_path + 'circle_figure8_fullset/'
model_path = trajectory_path + 'models/'

disturbance_path = trajectory_path + 'disturbance_full.npy'
input_path = trajectory_path + 'input_full.npy'
key = random.PRNGKey(2)
horizon = 300 #200
dt = 0.05 #0.01

# Custom gradient descent
grad_clip = 1
custom_gd_lr_rate = 0.01

# jaxopt 
lr_rate = 0.01

# scipy optimizer
iter_adam=1000 #4000

def initialize_sigma_points(X):
    '''
    Returns Equally weighted Sigma Particles
    '''
    n = X.shape[0]
    num_points = 2*n + 1
    X = X.reshape(-1,1)
    sigma_points = np.repeat( X, num_points, axis=1 )
    weights = np.ones((1,num_points)) * 1.0/( num_points )
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
def predict_states(X, policy_params):

    states = jnp.zeros((X.shape[0], horizon+1))
    states = states.at[:,0].set( X[:,0] )
    states_ref = jnp.zeros((X.shape[0], horizon))

    def body(h, inputs):
        '''
        Performs UT-EC with 6 states
        '''
        t = h * dt
        states, states_ref = inputs

        # Caclulates input with geometric controller
        control_input, pos_ref, vel_ref = policy( t, states[:,[h]], policy_params )         # mean_position = get_mean( states, weights )
        next_states, _ = get_next_states_ideal( states[:,[h]], control_input, dt )
        states = states.at[:,h+1].set( next_states[:,0] )
        states_ref = states_ref.at[:,h].set( jnp.append(pos_ref[:,0], vel_ref[:,0]) )
        return states, states_ref
    
    states =  lax.fori_loop( 0, horizon, body, (states, states_ref) )
    return states

def setup_future_reward_func():

    @jit
    def compute_reward(X, policy_params, gp_train_x, gp_train_y):
        '''
        Performs Gradient Descent
        '''
        states, weights = initialize_sigma_points(X)
        reward = 0

        # n = 6
        # N = 2*n+1 = 13
        # 13 becomes 169 pionts during expansion
        # compress back to 13 points

        def body(h, inputs):
            '''
            Performs UT-EC with 6 states
            '''
            t = h * dt
            reward, states, weights = inputs

            # Caclulates input with geometric controller
            control_inputs, pos_ref, vel_ref = policy( t, states, policy_params )         # mean_position = get_mean( states, weights )
            # jax.debug.print("🤯 states {x} 🤯", x=states)
            # jax.debug.print("🤯 inputs {x} 🤯", x=control_inputs)
            next_states_mean, next_states_cov = get_next_states_ideal( states, control_inputs, dt )
            ############################
            ####### bug fix: ##############
            ####### reshape cov to (6x7) #######
            #next_states_cov = next_states_cov.reshape(6,-1)
            # Expansion operation
            next_states_expanded, next_weights_expanded = sigma_point_expand_with_mean_cov( next_states_mean, next_states_cov, weights)
            
            # Compression operation
            next_states, next_weights = sigma_point_compress( next_states_expanded, next_weights_expanded )

            states = next_states
            weights = next_weights

            reward = reward + reward_func( states, weights, pos_ref, vel_ref ) # reward is loss
            return reward, states, weights
        
        reward =  lax.fori_loop( 0, horizon, body, (reward, states, weights) )[0]
        return reward
    return compute_reward
print(model_path)
file_path1 = model_path + "gp_model_x_norm5_full.pkl"
file_path2 = model_path + "gp_model_y_norm5_full.pkl"
file_path3 = model_path + "gp_model_z_norm5_full.pkl"
get_future_reward = setup_future_reward_func() #file_path1, file_path2, file_path3)
get_future_reward_grad = jit(grad(get_future_reward, argnums=1))




gp_train_x = jnp.load(input_path)
gp_train_x = gp_train_x[::80]
gp_train_y = jnp.load(disturbance_path)
gp_train_y = gp_train_y[::80].T
t0 = time.time()
# print(f"hello0 ")
# print(get_future_reward(jnp.zeros((6,1)), jnp.ones(2),gp_train_x, gp_train_y ))
# print(f"hello1 {time.time()-t0}")

# t0 = time.time()
# print(f"hello0 ")
# print(get_future_reward(jnp.zeros((6,1)), jnp.ones(2),gp_train_x, gp_train_y ))
# print(f"hello1 {time.time()-t0}")

# # t0 = time.time()
# # print(f"hello0 ")
# # print(get_future_reward(jnp.zeros((6,1)), jnp.ones(2),gp_train_x, gp_train_y ))
# # print(f"hello1 {time.time()-t0}")

# print(f"GRAD: ")
# t0 = time.time()
# print(f"hello0 ")
# print(get_future_reward_grad(jnp.zeros((6,1)), jnp.ones(2),gp_train_x, gp_train_y ))
# print(f"hello1 {time.time()-t0}")

# t0 = time.time()
# print(f"hello0 ")
# print(get_future_reward_grad(jnp.zeros((6,1)), jnp.ones(2),gp_train_x, gp_train_y ))
# print(f"hello1 {time.time()-t0}")

# t0 = time.time()
# print(f"hello0 ")
# print(get_future_reward_grad(jnp.zeros((6,1)), jnp.ones(2),gp_train_x, gp_train_y ))
# print(f"hello1 {time.time()-t0}")


def train_policy_jaxscipy(init_state, params_policy, gp_train_x, gp_train_y):
    minimize_func = lambda params: get_future_reward( init_state, params, gp_train_x, gp_train_y )
    solver = jaxopt.ScipyMinimize(fun=minimize_func, maxiter=iter_adam)
    params_policy, cost_state = solver.run(params_policy)
    return params_policy

def train_policy_custom(init_state, params_policy, gp_train_x, gp_train_y):
    for i in range(iter_adam):
        param_policy_grad = get_future_reward_grad( init_state, params_policy, gp_train_x, gp_train_y)
        param_policy_grad = np.clip( param_policy_grad, -grad_clip, grad_clip )
        params_policy = params_policy - custom_gd_lr_rate * param_policy_grad
        # params_policy =  np.clip( params_policy, -10, 10 )
def generate_state_vector(key, n):
    return jax.random.normal(key, (n, 1))

# Example usage:
key = jax.random.PRNGKey(0)  # Initialize the random key
n = 6  # Size of the state vector
state_vector = generate_state_vector(key, n)
print(state_vector)
# train_policy_custom(state_vector,  jnp.array([14.0, 7.4]),gp_train_x, gp_train_y)

# plot trajectory with initial parameter
# params_init = jnp.array([14.0, 7.4])
params_init = jnp.array([14.0, 7.4])
states, states_ref = predict_states(state_vector, params_init)
fig, ax = plt.subplots()
ax.plot(states_ref[0,:], states_ref[1,:], 'r', label='reference')
ax.plot(states[0,:], states[1,:], 'g', label='states unoptimized')
ax.set_xlim([-0.4,1.3])
ax.set_ylim([-1.3, 0.4])

ax.set_xlabel('X')
ax.set_ylabel('Y')
# plt.show()

params_optimized = train_policy_jaxscipy(state_vector, params_init, gp_train_x, gp_train_y)

states_optimized, states_ref_optimized = predict_states(state_vector, params_optimized)
ax.plot(states_optimized[0,:], states_optimized[1,:], 'k', label='states optimized')
ax.legend()

plt.show()





