import numpy as np
import matplotlib.pyplot as plt

# Define error signal e
e = np.linspace(-5, 5, 400)

# Control parameters
U_max = 0.2*9.81  # Maximum control value
# k_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]  # Different k values for illustration
# k_values = np.linspace(0.01,1,10 )
k_values = [3]
# Plot the control output for different k values
plt.figure(figsize=(12, 8))

for k in k_values:
    u = U_max * np.tanh(k * e)
    plt.plot(e, u, label=f'$k = {k}$')

# plt.title('Effect of Different k Values on tanh-based Control')
plt.title('tanh with slope = 0.5097')
plt.xlabel('Error Signal (e)')
plt.ylabel('Control Output (u)')
plt.grid(True)
# plt.ylim(-1,1)
plt.legend()
plt.show()
