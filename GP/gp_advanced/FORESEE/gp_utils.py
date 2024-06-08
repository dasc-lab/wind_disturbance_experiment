import jax
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



def get_next_states_with_gp( states, control_inputs, gps ):
    states_gp = np.concatenate( (states[0,:].reshape(1,-1), states[1,:].reshape(1,-1), states[3,:].reshape(1,-1), np.sin(states[2,:]).reshape(1,-1), np.cos(states[2,:]).reshape(1,-1)), axis=0 )
    test_x = np.append( states_gp, control_inputs.reshape(1,-1), axis=0 ).T
    dt = 0.05
    
    # gp gives differences only
    pred2 = gps[1](test_x)
    mu2, var2 = pred2.mean(), pred2.variance()
    
    pred4 = gps[3](test_x)
    mu4, var4 = pred4.mean(), pred4.variance()
    
    # pred1 = gps[0](test_x)
    # mu1, var1 = pred1.mean(), pred1.variance()
    # mu1, var1 = np.array([ states[0,:] + dt * states[1,:]  ]), np.zeros((1,9))
    mu1, var1 = dt / 2 * states[1,:] + dt / 2 * (states[1,:]+mu2), np.zeros((1,9))
    
    # pred3 = gps[2](test_x)
    # mu3, var3 = pred3.mean(), pred3.variance()
    # mu3, var3 = np.array([ wrap_angle(states[2,:] + dt * states[3,:])  ]), np.zeros((1,9))
    mu3, var3 = dt / 2 * states[3,:] + dt / 2 * (states[3,:]+mu4), np.zeros((1,9))
    
    return states+np.concatenate((mu1.reshape(1,-1), mu2.reshape(1,-1), mu3.reshape(1,-1), mu4.reshape(1,-1)), axis=0), np.concatenate( (var1.reshape(1,-1), var2.reshape(1,-1), var3.reshape(1,-1), var4.reshape(1,-1)), axis=0 )
    

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

def initialize_gp_prediction(gp_params, train_x, train_y):
    def load_gp_model(file_path):
        with open(file_path, 'rb') as f:
            opt_posterior = pickle.load(f)
        return opt_posterior
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

def initialize_gp_prediction(file_path, gp_params, train_x, train_y, test_x):
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