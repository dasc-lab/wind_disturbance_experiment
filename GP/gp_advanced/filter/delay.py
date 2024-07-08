import numpy as np
from scipy.signal import butter, group_delay
import matplotlib.pyplot as plt

def calculate_group_delay(order, cutoff_frequency, sampling_rate):
    # Design a Butterworth lowpass filter
    b, a = butter(order, cutoff_frequency, btype='low', fs=sampling_rate)
    
    # Calculate the group delay
    w, delay = group_delay((b, a), fs=sampling_rate)
    
    return w, delay

# Parameters
order = 1  # Filter order
sampling_rate = 5000  # Sampling rate
cutoff_frequencies = np.linspace(200, 2000, 100)  # Range of cutoff frequencies
sampling_rates = np.linspace(100.0, 10000.0, 100)
sampling_rates = [300, 500, 1000, 2000, 5000] 
sampling_rates = [60, 80, 100, 125, 150] 
# Store the maximum group delay for each cutoff frequency
max_group_delays = []

for cutoff_frequency in cutoff_frequencies:
    w, delay = calculate_group_delay(order, cutoff_frequency, sampling_rate)
    max_group_delays.append(np.max(delay))

# Plotting the group delay with respect to cutoff frequency
plt.figure()
plt.plot(cutoff_frequencies, max_group_delays, marker='o')
plt.title('Group Delay with Respect to Cutoff Frequency')
plt.xlabel('Cutoff Frequency')
plt.ylabel('Maximum Group Delay [samples]')
plt.grid()
plt.show()

plt.figure(figsize=(12,8))
for sr in sampling_rates:
    w, delay = calculate_group_delay(order, 20, sr)
    plt.plot(w, delay, label=f'Sampling Rate = {sr} Hz')
plt.title('Group Delay of Butterworth Lowpass Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Group Delay (samples)')
plt.legend()
plt.grid()
plt.show()


# Define the Butterworth filters
def butter_lowpass(cutoff, fs, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Sampling frequency
fs1 = 20.0  # Hz
fs2 = 12
# Cutoff frequencies for the two filters
cutoff1 = 4.6  # Hz
cutoff2 = 4  # Hz

# Filter order
order = 1

# Get the filter coefficients
b1, a1 = butter_lowpass(cutoff1, fs1, order)
b2, a2 = butter_lowpass(cutoff2, fs2, order)

w1, delay1 = group_delay((b1,a1),fs=fs1)
w2, delay2 = group_delay((b2,a2),fs=fs2)

# delay1 = delay1/fs1
# delay2 = delay2/fs2

plt.figure(figsize=(10, 6))
plt.plot(w1, delay1, label=f'Butterworth filter with cutoff = {cutoff1} Hz')
plt.plot(w2, delay2, label=f'Butterworth filter with cutoff = {cutoff2} Hz')
plt.xlabel('Frequency [Hz]')  # X-axis in Hertz
plt.ylabel('Group delay [seconds]')  # Y-axis in seconds
plt.title('Group Delay of Butterworth Lowpass Filters')
plt.legend()
plt.grid()
plt.show()
