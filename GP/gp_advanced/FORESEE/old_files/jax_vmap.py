import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
# Define a simple function that adds two numbers
def add(x, y):
    return x + y

# Vectorize the add function using vmap
vectorized_add = jax.vmap(add, in_axes=(0, None))

# Define two arrays
array1 = jnp.array([1, 2, 3, 4])
array2 = jnp.array([5, 6, 7, 8])
num = 1
# Apply the vectorized function to the arrays
result = vectorized_add(array1, num)

# Print the result
print("Result of vectorized addition:", result)

import jax.numpy as jnp

# Create an empty array with shape (0, 3)
empty_array = jnp.empty((0, 3))

# Create a (51, 3) JAX array
key = jrandom.PRNGKey(0)  # PRNG key for random number generation
array = jrandom.uniform(key, (51, 3))  # Generate a (51, 3) array of random numbers

# Vertically stack the arrays
stacked_array = jnp.vstack((empty_array, array))
print(stacked_array)