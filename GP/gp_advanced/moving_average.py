import numpy as np
import matplotlib.pyplot as plt

# Generate sample data for a 3D signal

data = np.load("recorded_acc.npy")

# Define the moving average filter
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

# Apply the moving average filter to each column

window_size = 10
filtered_data = np.array([moving_average(data[:, i], window_size) for i in range(3)]).T

# Plot the original and filtered data for each column in separate figures
for i in range(3):
    plt.figure(figsize=(10, 5))
    plt.plot(data[:, i], label='Original')
    plt.plot(np.arange(window_size - 1, len(data)), filtered_data[:, i], label='Filtered')
    plt.title(f'Column {i+1}')
    plt.legend()
    plt.show()
