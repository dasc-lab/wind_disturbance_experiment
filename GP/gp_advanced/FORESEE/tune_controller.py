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

from jax_utils import *
from gp_utils import *
from policy import policy

key = random.PRNGKey(2)
horizon = 30
dt = 0.01

# Custom gradient descent
grad_clip = 1
custom_gd_lr_rate = 0.01

# jaxopt 
lr_rate = 0.01

# scipy optimizer
iter_adam=4000

def initialize_sigma_points(X):
    '''
    Returns Equally weighted Sigma Particles
    '''
    # return 2N + 1 points
    n = X.shape[0]
    num_points = 2*n + 1
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
    reward = pos_factor * jnp.sum(ex_ev_mean[0:3] ** 2) + vel_factor * jnp.sum(ex_ev_mean[3:6])
    return reward

def setup_future_reward_func(file_path1, file_path2, file_path3):

    gp0 = initialize_gp_prediction( file_path1 ) #, gp_train_x, gp_train_y[:,0].reshape(-1,1) )
    gp1 = initialize_gp_prediction( file_path2 ) #, gp_train_x, gp_train_y[:,1].reshape(-1,1) )
    gp2 = initialize_gp_prediction( file_path3 ) #, gp_train_x, gp_train_y[:,2].reshape(-1,1) )

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
            next_states_mean, next_states_cov = get_next_states_with_gp( states, control_inputs, [gp0, gp1, gp2], gp_train_x, gp_train_y )

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

# file_path1 = ""
# file_path2 = ""
# file_path3 = ""
get_future_reward = setup_future_reward_func(file_path1, file_path2, file_path3)
get_future_reward_grad = jit(grad(get_future_reward, 1))

gp_train_x = ""
gp_train_y = ""





def train_policy( run, key, use_custom_gd, use_jax_scipy, use_adam, adam_start_learning_rate, init_state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y ):
    '''
    Three Potential Candidates for optimizer, custom Gradient descent, jax scipy optimizer, and adam optimizer. learning rate needs to be adjusted
    '''
# if (optimize_offline):
#     #train using scipy ###########################

    if use_custom_gd:
        for i in range(100):
            param_policy_grad = get_future_reward_grad( init_state, params_policy, gp_train_x, gp_train_y)
            param_policy_grad = np.clip( param_policy_grad, -grad_clip, grad_clip )
            params_policy = params_policy - custom_gd_lr_rate * param_policy_grad
            # params_policy =  np.clip( params_policy, -10, 10 )
        print(f"reward final GD : { get_future_reward( init_state, params_policy, gp_train_x, gp_train_y ) }")

    if use_jax_scipy:
        costs_adam = []
        minimize_function = lambda params: get_future_reward( init_state, params, gp_train_x, gp_train_y )
        solver = jaxopt.ScipyMinimize(fun=minimize_function, maxiter=iter_adam)
        params_policy, cost_state = solver.run(params_policy)
        print(f"Jaxopt state: {cost_state}")

    if use_adam:
        # print(f"inside adam")
        # t0 = time.time()
        cost = get_future_reward( init_state, params_policy, gp_train_x, gp_train_y )
        # print(f"adam first reward: {time.time()-t0}")
        best_params = np.copy(params_policy)
        best_cost = np.copy(cost)
        costs_adam = []
        
        cost = get_future_reward( init_state, params_policy, gp_train_x, gp_train_y )
        cost_run = [cost]
        
        scheduler = optax.exponential_decay(
            init_value=lr_rate, 
            transition_steps=100,
            decay_rate=0.95)

        # Combining gradient transforms using `optax.chain`.
        gradient_transform = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
            optax.scale_by_adam(),  # Use the updates from adam.
            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )
        opt_state = gradient_transform.init(params_policy)
        reset = 0
        params_policy_temp = 0
        for i in range(iter_adam + 1):
            reset = reset + 1
            
            grads = get_future_reward_grad( init_state, params_policy, gp_train_x, gp_train_y)
            updates, opt_state = gradient_transform.update(grads, opt_state)                
            params_policy = optax.apply_updates(params_policy, updates)
            
            # book keeping
            cost = get_future_reward( init_state, params_policy, gp_train_x, gp_train_y)
            cost_run.append(cost)
        
        params_policy = np.copy(best_params)
            
        with open('new_rl.npy', 'wb') as f:
            np.save(f, best_params)    
        print(f" *************** NANs? :{np.any(np.isnan(params_policy)==True)} ")
    return key, params_policy, costs_adam


