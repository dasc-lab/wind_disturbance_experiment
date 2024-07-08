import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import optax as ox
import gpjax as jgp
import pickle
import jaxkern as jk
from gpjax import Dataset
import jax.numpy as np
from jax import jit, random
from gpjax.kernels import SumKernel, White, RBF, Matern32, RationalQuadratic, Periodic, ProductKernel
key = random.PRNGKey(2)
home_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/FORESEE/'



def get_next_states_with_gp( states, control_inputs, gps, train_x, train_y, dt ):

    # latent_dist = opt_posterior(test_x, D)
    # predictive_dist = opt_posterior.likelihood(latent_dist)
    # pred_mean = predictive_dist.mean()
    # pred_std = predictive_dist.stddev()

    normalize_factor = 3

    # pred_mean = pred_mean*normalize_factor
    # pred_std = pred_std*normalize_factor

    # For GP, each input should be a row vector
    test_x = states.T #jnp.append( states.T, control_inputs.T, axis=0)

    # X disturbance
    D = Dataset(X=train_x, y=train_y[0])
    latent_dist = gps[0](test_x, D)
    predictive_dist = gps[0].likelihood(latent_dist)
    pred_mean0 = predictive_dist.mean().reshape(-1,1)
    pred_std0 = predictive_dist.stddev().reshape(-1,1)

    # Y disturbance
    D = Dataset(X=train_x, y=train_y[1])
    latent_dist = gps[1](test_x, D)
    predictive_dist = gps[1].likelihood(latent_dist)
    pred_mean1 = predictive_dist.mean().reshape(-1,1)
    pred_std1 = predictive_dist.stddev().reshape(-1,1)

    # Z disturbance
    D = Dataset(X=train_x, y=train_y[2])
    latent_dist = gps[2](test_x, D)
    predictive_dist = gps[2].likelihood(latent_dist)
    pred_mean2 = predictive_dist.mean().reshape(-1,1)
    pred_std2 = predictive_dist.stddev().reshape(-1,1)

    pred_mu = jnp.concatenate( (pred_mean0, pred_mean1, pred_mean2), axis=0 )
    pred_cov = jnp.concatenate( (pred_std0**2, pred_std1**2, pred_std2**2), axis=0 )

    next_states_pos = states[0:3] + states[3:6] * dt
    next_states_vel_mu = states[3:6] + control_inputs * dt + pred_mu * dt
    next_states_vel_cov = pred_cov * dt * dt

    next_states_mu = jnp.append( next_states_pos, next_states_vel_mu, axis=0 )
    next_states_cov = jnp.append( jnp.zeros((3,1)), next_states_vel_cov, axis=0 )
    return next_states_mu, next_states_cov    

def initialize_gp(num_datapoints = 10):
    meanf = jgp.mean_functions.Zero()
    # kernel = jk.RBF(active_dims=[0,1,2,3,4]) * jk.Polynomial(active_dims=[0,1,2,3,4])
    # kernel = jk.RBF(active_dims=[0,1,2,3,4]) * jk.RationalQuadratic(active_dims=[0,1,2,3,4])
    kernel = jk.RBF(active_dims=[0,1,2,3,4,5]) #* jk.Periodic(active_dims=[0,1,2,3,4])
    # kernel = jk.RBF() #* jk.Periodic()
    prior = jgp.Prior(mean_function=meanf, kernel = kernel)
    likelihood = jgp.Gaussian( num_datapoints=num_datapoints )
    posterior = prior * likelihood
    parameter_state = jgp.initialise(
        posterior, key#, kernel={"lengthscale": np.array([0.5])}
    )
    return likelihood, posterior, parameter_state

def train_gp(likelihood, posterior, parameter_state, train_x, train_y):
    D = Dataset(X=train_x, y=train_y)
    negative_mll = jit(posterior.marginal_log_likelihood(D, negative=True))
    optimiser = ox.adam(learning_rate=0.08)
    inference_state = jgp.fit(
        objective=negative_mll,
        parameter_state=parameter_state,
        optax_optim=optimiser,
        num_iters=1500,
    )
    learned_params, training_history = inference_state.unpack()
    return likelihood, posterior, learned_params, D

def predict_gp(likelihood, posterior, learned_params, D, test_x):
    latent_dist = posterior(learned_params, D)(test_x)
    predictive_dist = likelihood(learned_params, latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()
    return predictive_mean, predictive_std

def initialize_gp_prediction_distribution(gp_params, train_x, train_y):
    meanf = jgp.mean_functions.Zero()
    # kernel = jk.RBF(active_dims=[0,1,2,3,4]) #* jk.Polynomial(active_dims=[0,1,2,3,4])
    # kernel = jk.RBF(active_dims=[0,1,2,3,4]) * jk.RationalQuadratic(active_dims=[0,1,2,3,4])
    kernel = jk.RBF(active_dims=[0,1,2,3,4,5]) #* jk.Periodic(active_dims=[0,1,2,3,4])
    # kernel = jk.RBF() #* jk.Periodic()
    prior = jgp.Prior(mean_function=meanf, kernel = kernel)
    likelihood = jgp.Gaussian( num_datapoints=train_x.shape[0] )
    posterior = prior * likelihood
    D = Dataset(X=train_x, y=train_y)
    latent_dist = posterior(gp_params, D)
    return latent_dist

# def initialize_gp_prediction(gp_params, train_x, train_y):
#     def load_gp_model(file_path):
#         with open(file_path, 'rb') as f:
#             opt_posterior = pickle.load(f)
#         return opt_posterior
#     meanf = jgp.mean_functions.Zero()
#     # kernel = jk.RBF(active_dims=[0,1,2,3,4]) #* jk.Polynomial(active_dims=[0,1,2,3,4])
#     # kernel = jk.RBF(active_dims=[0,1,2,3,4]) * jk.RationalQuadratic(active_dims=[0,1,2,3,4])
#     kernel = jk.RBF(active_dims=[0,1,2,3,4,5]) #* jk.Periodic(active_dims=[0,1,2,3,4])
#     # kernel = jk.RBF() #* jk.Periodic()
#     prior = jgp.Prior(mean_function=meanf, kernel = kernel)
#     likelihood = jgp.Gaussian( num_datapoints=train_x.shape[0] )
#     posterior = prior * likelihood
#     D = Dataset(X=train_x, y=train_y)
#     latent_dist = posterior(gp_params, D)
#     return latent_dist

def initialize_gp_prediction(file_path):#, train_x, train_y): #, test_x):
    def load_gp_model(file_path):
        with open(file_path, 'rb') as f:
            opt_posterior = pickle.load(f)
        return opt_posterior

    # latent_dist = posterior(gp_params, D)
    # D = Dataset(X=train_x, y=train_y)
    opt_posterior = load_gp_model(file_path)
    return opt_posterior
    latent_dist = opt_posterior(test_x, D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    normalize_factor = 3
# Obtain the predictive mean and standard deviation
    pred_mean = predictive_dist.mean()
    pred_std = predictive_dist.stddev()

    pred_mean = pred_mean*normalize_factor
    pred_std = pred_std*normalize_factor
    
    return latent_dist

def predict_with_gp_params(gp_params, train_x, train_y, test_x):
    meanf = jgp.mean_functions.Zero()
    # kernel = jk.RBF(active_dims=[0,1,2,3,4]) * jk.Polynomial(active_dims=[0,1,2,3,4])
    # kernel = jk.RBF(active_dims=[0,1,2,3,4]) * jk.RationalQuadratic(active_dims=[0,1,2,3,4])
    kernel = jk.RBF(active_dims=[0,1,2,3,4,5]) #* jk.Periodic(active_dims=[0,1,2,3,4])
    # kernel = jk.RBF() #* jk.Periodic()
    prior = jgp.Prior(mean_function=meanf, kernel = kernel)
    likelihood = jgp.Gaussian( num_datapoints=train_x.shape[0] )
    posterior = prior * likelihood
    D = Dataset(X=train_x, y=train_y)
    latent_dist = posterior(gp_params, D)(test_x)
    predictive_dist = likelihood(gp_params, latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_var = predictive_dist.stddev()**2
    return predictive_mean, predictive_var


def predict_with_gp_params(file_path, gp_params, train_x, train_y, test_x):
    def load_gp_model(file_path):
        with open(file_path, 'rb') as f:
            opt_posterior = pickle.load(f)
        return opt_posterior
    # meanf = jgp.mean_functions.Zero()
    # # kernel = jk.RBF(active_dims=[0,1,2,3,4]) #* jk.Polynomial(active_dims=[0,1,2,3,4])
    # # kernel = jk.RBF(active_dims=[0,1,2,3,4]) * jk.RationalQuadratic(active_dims=[0,1,2,3,4])
    # kernel = jk.RBF(active_dims=[0,1,2,3,4,5]) #* jk.Periodic(active_dims=[0,1,2,3,4])
    # # kernel = jk.RBF() #* jk.Periodic()
    # prior = jgp.Prior(mean_function=meanf, kernel = kernel)
    # likelihood = jgp.Gaussian( num_datapoints=train_x.shape[0] )
    # posterior = prior * likelihood
    
    # latent_dist = posterior(gp_params, D)
    D = Dataset(X=train_x, y=train_y)
    opt_posterior = load_gp_model(file_path)
    latent_dist = opt_posterior(test_x, D)
    
    predictive_dist = opt_posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_var = predictive_dist.stddev()**2
    return predictive_mean, predictive_var