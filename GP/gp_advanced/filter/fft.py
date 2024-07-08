#######################################################################################################
####### Use this file to load any npy file and analyze its frequency domain counterpart           #####
####### The file also automatically filters out the high-frequency noise                          #####
####### NOTE: the low pass filter calculates cutoff frequency by adding a hard-coded value to the #####
####### frequency of the highest peak in frequency domain. Adjust this value if needed            #####
#######################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt, group_delay
import os, sys
path = os.path.join('/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/')

sys.path.append(path)
from plot_trajectory_ref import bag_path, cutoff, threshold

signal = np.load("recorded_acc.npy")
#signal = np.load("cmd_arr.npy")
#signal = np.load("disturbance.npy")
cutoff = signal.shape[0]
threshold = 0
signal = signal[:,0][threshold:cutoff]
N = len(signal)

print("signal shape = ", signal.shape)
t = np.arange(signal.shape[0])
sampling_rate = fs = 20#2000
yf = fft_signal = np.fft.fft(signal)
xf = fft_freq = np.fft.fftfreq(len(signal), 1 / sampling_rate)[:len(fft_signal)//2]


magnitude = 2.0/N * np.abs(yf[:N//2])

# Find the peak frequency
peak_index = np.argmax(magnitude)
peak_frequency = xf[peak_index]
peak_amplitude = magnitude[peak_index]

print(f"Peak frequency: {peak_frequency} Hz")
print(f"Peak amplitude: {peak_amplitude}")
# Plot the original signal
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Time Domain Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plot the FFT (magnitude spectrum)
plt.subplot(2, 1, 2)
# plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_signal)[:len(fft_signal)//2] * 2 / len(signal))
plt.plot(fft_freq, np.abs(fft_signal)[:len(fft_signal)//2] * 2 / len(signal))
plt.title("Frequency Domain Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()

def butter_lowpass_filter(data, cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y , b , a

cutoff_freq = peak_frequency+4.0#Hz
filtered_signal = filtered_data = butter_lowpass_filter(signal, cutoff_freq, fs, order = 1)[0]
print("cutoff frequency: ", cutoff_freq)
print("sampling frequency", fs)
_,b,a = butter_lowpass_filter(signal, cutoff_freq, fs, order = 1)
plt.figure(figsize=(12, 6))
plt.plot(signal, label='Original Signal')
plt.plot(filtered_data, label='Filtered Signal', linestyle='--')
plt.title(f'Original and Low-pass Filtered Signal cutoff = {cutoff_freq}, sr = {fs}')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()


plt.figure()
w, delay = group_delay((b,a), fs=fs)
plt.plot(w,delay)
plt.xlabel('Frequency [Hz]')  # X-axis in Hertz
plt.ylabel('Group delay')  # Y-axis in seconds
plt.title('Group Delay of Butterworth Lowpass Filter')
plt.show()
#np.save("filtered_y_component.npy", filtered_signal)

def fft_filter(signal, sampling_rate = 20):
    yf = fft_signal = np.fft.fft(signal)
    xf = fft_freq = np.fft.fftfreq(len(signal), 1 / sampling_rate)[:len(fft_signal)//2]
    N = len(signal)

    magnitude = 2.0/N * np.abs(yf[:N//2])

    # Find the peak frequency
    peak_index = np.argmax(magnitude)
    peak_frequency = xf[peak_index]
    peak_amplitude = magnitude[peak_index]
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    ####### NOTE: Replace  if necessary to calculate the cutoff frequency ######
    cutoff_freq = peak_frequency+2 #Hz
    filtered_signal = filtered_data = butter_lowpass_filter(signal, cutoff_freq, fs, order=2)
    return filtered_signal