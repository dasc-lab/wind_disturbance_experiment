import jax.numpy as jnp







def kernel(x,y):
    return x, y

def prior_mean_function(x):
      return x


def prior_kernel_gram(x):
      return


def prior_kernel_cross_covariance(x, t):
      return

def predict_with_sigma_inv(
        test_inputs,
        train_data,
        Sigma_inv,
        prior_jitter,
    ):
        
        # Unpack training data
        x, y = train_data.X, train_data.y

        # Unpack test inputs
        t = test_inputs

        # Observation noise o²
        # obs_noise = self.likelihood.obs_stddev**2
        mx = prior_mean_function(x)

        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = prior_kernel_gram(x)
        Kxx += cola.ops.I_like(Kxx) * self.jitter

        # Σ = Kxx + Io²
        # Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
        # Sigma = cola.PSD(Sigma)

        mean_t = prior_mean_function(t)
        Ktt = prior_kernel_gram(t)
        Kxt = prior_kernel_cross_covariance(x, t)

        # Sigma_inv_Kxt = cola.solve(Sigma, Kxt)
        Sigma_inv_Kxt = Sigma_inv @ Kxt

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += cola.ops.I_like(covariance) * prior_jitter
        covariance = cola.PSD(covariance)

        return conac