from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
#from geometry_msgs.msg import TransformStamped
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
plotter_path = os.path.join('/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/')
sys.path.append(plotter_path)
from plot_trajectory_ref import bag_path, cutoff, threshold
from scipy.fftpack import fft, fftfreq

from scipy.signal import butter, filtfilt

print("bag path is: ", bag_path)
print("cutoff = ", cutoff)
print("threshold = ", threshold)
def fft_filter(signal, sampling_rate = 5000):
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

    cutoff_freq = peak_frequency+50 #Hz
    filtered_signal = filtered_data = butter_lowpass_filter(signal, cutoff_freq, sampling_rate)
    return filtered_signal

def apply_fft_filter_to_columns(array, sampling_rate=5000):
    filtered_array = np.zeros_like(array)
    for i in range(array.shape[1]):
        filtered_array[:, i] = fft_filter(array[:, i], sampling_rate)
    return filtered_array
topic_name = '/drone/combined_data' # Add more topics as needed
typestore = get_typestore(Stores.LATEST)

msg_text = Path("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/DynamicsData.msg").read_text()
add_types = {}
# Add definitions from one msg file to the dict.
add_types.update(get_types_from_msg(msg_text, 'foresee_msgs/msg/DynamicsData'))
typestore.register(add_types)
kx = 14
kv = 7.4
m = 0.681
g = 9.81
pos_arr = []

vel_arr = []
acc_cmd_arr = []
acc_arr = []
acc_ref = 0
acc_ref_arr = []
disturbance = []

with Reader(bag_path) as reader:
    for item in reader.connections:
        print("topic, message type:",item.topic, item.msgtype)
    for item, timestamp, rawdata in reader.messages():
        if item.topic == topic_name:
            msg = typestore.deserialize_cdr(rawdata, item.msgtype)
            #print(type(msg.pos))
            pos_arr.append(msg.pos)
            vel_arr.append(msg.vel)
            pos = msg.pos
            vel = msg.vel
            acc = msg.acc
            pos_ref = msg.pos_ref
            vel_ref = msg.vel_ref
            acc_ref = msg.acc_ref
            diff_pos = pos - pos_ref
            diff_vel = vel - vel_ref
            thrust = -kx*diff_pos -kv*diff_vel + m * acc_ref #- g*m 
            acc_ref_arr.append(acc_ref)
            #thrust = thrust  + g * m
            acc_cmd = thrust/m
            acc_cmd_arr.append(acc_cmd)
            acc_arr.append(acc)
            acc_diff = acc - acc_cmd
            #disturbance.append(acc_diff)

acc_arr = np.array(acc_arr)
#np.save("recorded_acc.npy",acc_arr)
filtered_acc = apply_fft_filter_to_columns(acc_arr, sampling_rate=5000)
acc_cmd_arr = np.array(acc_cmd_arr)
#np.save("cmd_arr.npy",acc_cmd_arr)
filtered_cmd = apply_fft_filter_to_columns(acc_cmd_arr, sampling_rate=5000)
filtered_cmd = acc_cmd_arr
#disturbance = filtered_acc - acc_cmd_arr
disturbance = filtered_acc - filtered_cmd
assert acc_arr.shape == acc_cmd_arr.shape == disturbance.shape == filtered_acc.shape
plot_size = acc_arr.shape[0]
# Create a function to plot the data
def plot_same_dim(array1, array2, array3, dim, title, figure_number):
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle(title)

    arrays = [array1, array2, array3]
    labels = ['act', 'cmd', 'disturbance']
    for i, array in enumerate(arrays):
        axs[i].plot(array[:, dim])
        axs[i].set_title(labels[i])
        axs[i].set_xlabel('Time Index')
        axs[i].set_ylabel(title[-1])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top margin for the suptitle
    plt.savefig(f'figure_{figure_number}.png')
    plt.show()

# Plot each dimension in a separate figure
plot_same_dim(filtered_acc[threshold:cutoff], filtered_cmd[threshold:cutoff], disturbance[threshold:cutoff], 0, 'acc_x', 1)
plot_same_dim(filtered_acc[threshold:cutoff], filtered_cmd[threshold:cutoff], disturbance[threshold:cutoff], 1, 'acc_y', 2)
plot_same_dim(filtered_acc[threshold:cutoff], filtered_cmd[threshold:cutoff], disturbance[threshold:cutoff], 2, 'acc_z', 3)
# for i in range(3):
#     fig = plt.figure(i+1)
#     ax1 = plt.subplot2grid((3,1), (0,0) , rowspan=1)
#     ax1.plot(range(plot_size), acc_ref[:,i] , 'r',label='Actual trajectory')
#     ax1.legend(loc='upper right')
#     ax1.set_title("x")
#     ax1.set_ylabel("meters")
#     ax1 = plt.subplot2grid((3,1), (1,0) , rowspan=1)
#     ax1.plot(range(y.size), y,'r',label='Actual trajectory')
#     ax1.plot(range(y_t.size), y_t.T, 'b',label='Reference trajectory')
#     ax1.legend(loc='upper right')
#     ax1.set_title("y")
#     ax1.set_ylabel("meters")

#     ax1 = plt.subplot2grid((3,1), (2,0) , rowspan=1)
#     ax1.plot(range(z.size), z,'r',label='Actual trajectory')
#     ax1.plot(range(z_t.size), z_t.T, 'b',label='Reference trajectory')
#     ax1.legend(loc='upper right')
#     ax1.set_ylabel("meters")
#     ax1.set_title("z")