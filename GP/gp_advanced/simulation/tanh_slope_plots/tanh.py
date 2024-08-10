import numpy as np
import matplotlib.pyplot as plt

# Define error signal e
e = np.linspace(-5, 5, 400)

# Control parameters
# U_max = 0.2*9.81  # Maximum control value
U_max = 6*0.681
# k_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]  # Different k values for illustration
# k_values = np.linspace(0.01,1,10 )
k_values = [0.286]
# Plot the control output for different k values
plt.figure(figsize=(12, 8))

for k in k_values:
    u = U_max * np.tanh(k * e)
    plt.plot(e, u, label=f'$k = {k}$')

# plt.title('Effect of Different k Values on tanh-based Control')
plt.title(f'tanh with slope = {k_values[0]}')
plt.xlabel('Input Acc (m/s^2)')
plt.ylabel('a*tanh(kx)')
plt.grid(True)
# plt.ylim(-1,1)
plt.legend()
plt.show()
