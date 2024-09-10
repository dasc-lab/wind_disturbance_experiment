from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
from scipy.signal import butter, filtfilt
'''
Helpers for filtering
'''
def fft_filter(signal, sampling_rate = 1000):
    yf = fft_signal = np.fft.fft(signal)
    xf = fft_freq = np.fft.fftfreq(len(signal), 1 / sampling_rate)[:len(fft_signal)//2]
    N = len(signal)

    magnitude = 2.0/N * np.abs(yf[:N//2])

    # Find the peak frequency
    peak_index = np.argmax(magnitude[0:10])
    peak_frequency = xf[peak_index]
    peak_amplitude = magnitude[peak_index]
    
    def butter_lowpass_filter(data, cutoff_freq, fs, order=1):
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    cutoff_freq = peak_frequency+3.0 
    
    filtered_signal = filtered_data = butter_lowpass_filter(signal, cutoff_freq, sampling_rate)
    return filtered_signal
def apply_fft_filter_to_columns(array, sampling_rate=1000):
    filtered_array = np.zeros_like(array)
    for i in range(array.shape[1]):
        filtered_array[:, i] = fft_filter(array[:, i], sampling_rate)
    return filtered_array


'''
Helpers for plotter
'''
def plot_helper(pos_vector, pos_ref_vector):
    x = pos_vector[:,0]
    x_t = pos_ref_vector[:,0]
    y = pos_vector[:,1]
    y_t = pos_ref_vector[:,1]
    z = pos_vector[:,2]
    z_t = pos_ref_vector[:,2]
    fig = plt.figure()
    ax1 = plt.subplot2grid((3,1), (0,0) , rowspan=1)
    ax1.plot(range(x.size), x , 'r',label='Actual trajectory')
    ax1.plot(range(x_t.size), x_t.T, 'b',label='Reference trajectory')
    ax1.legend(loc='upper right')
    ax1.set_title("x")
    ax1.set_ylabel("meters")
    ax1 = plt.subplot2grid((3,1), (1,0) , rowspan=1)
    ax1.plot(range(y.size), y,'r',label='Actual trajectory')
    ax1.plot(range(y_t.size), y_t.T, 'b',label='Reference trajectory')
    ax1.legend(loc='upper right')
    ax1.set_title("y")
    ax1.set_ylabel("meters")
    ax1 = plt.subplot2grid((3,1), (2,0) , rowspan=1)
    ax1.plot(range(z.size), z,'r',label='Actual trajectory')
    ax1.plot(range(z_t.size), z_t.T, 'b',label='Reference trajectory')
    ax1.legend(loc='upper right')
    ax1.set_ylabel("meters")
    ax1.set_title("z")
    plt.subplots_adjust(hspace=0.5) 
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos_vector[:,0],pos_vector[:,1],pos_vector[:,2], c = 'r')
    ax.scatter(pos_ref_vector[:,0],pos_ref_vector[:,1],pos_ref_vector[:,2], c = 'b')
    ax.set_zlim(0,1)
    plt.show()

def plot_trajectory(bag_path, takeoff, land):
    pos_vector = []
    pos_ref_vector = []
    vel_vector = []
    vel_ref_vector = []
    acc_vector = []
    acc_ref_vector = []
    quaternion_vector = []
    kx = 7
    kv = 4
    m = 0.681
    acc_cmd_arr = []
    acc_arr = []
    def initialize_typestore():
        typestore = get_typestore(Stores.LATEST)
        msg_text = Path('../DynamicsData.msg').read_text()
        add_types = {}
        add_types.update(get_types_from_msg(msg_text, 'foresee_msgs/msg/DynamicsData'))
        typestore.register(add_types)
        return typestore
    topic_name = '/drone/combined_data'
    typestore = initialize_typestore()
    with Reader(bag_path) as reader:
        for item in reader.connections:
            print(item.topic, item.msgtype)
        for item, timestamp, rawdata in reader.messages():
            if item.topic == topic_name:
                msg = typestore.deserialize_cdr(rawdata, item.msgtype)
                pos_vector.append(msg.pos)
                vel_vector.append(msg.vel)
                acc_vector.append(msg.acc)
                quaternion_vector.append(msg.quaternion)
            
                pos_ref_vector.append(msg.pos_ref)
                vel_ref_vector.append(msg.vel_ref)
                diff_pos = msg.pos - msg.pos_ref
                diff_vel = msg.vel - msg.vel_ref
                if norm(diff_pos) > 2:
                    diff_pos = 2*diff_pos/norm(diff_pos)
                if norm(diff_vel) > 5:
                    diff_vel = 5*diff_vel/norm(diff_vel)
                thrust = -kx*diff_pos -kv*diff_vel + m * msg.acc_ref #- g*m 
                #thrust = thrust  + g * m
                acc_cmd = thrust/m
                acc_cmd_arr.append(acc_cmd)
                acc_arr.append(msg.acc)
        
        pos_vector = np.array(pos_vector)
        pos_ref_vector = np.array(pos_ref_vector)
        vel_vector = np.array(vel_vector)
        vel_ref_vector = np.array(vel_ref_vector)
        acc_arr = np.array(acc_arr)
        acc_cmd_arr = np.array(acc_cmd_arr)
        quaternion_vector = np.array(quaternion_vector)
        gp_input = np.hstack((pos_vector, vel_vector, acc_cmd_arr, quaternion_vector))
        # gp_input = np.hstack((pos_vector, vel_vector, acc_cmd_arr))
        filtered_acc = apply_fft_filter_to_columns(acc_arr, sampling_rate=1000)
        assert filtered_acc.shape == acc_cmd_arr.shape
        disturbance = filtered_acc - acc_cmd_arr


        length = pos_vector.shape[0]
        pos_vector[:,2] = -pos_vector[:,2] # only for plotting
        pos_ref_vector[:,2] = -pos_ref_vector[:,2]
    return pos_vector[takeoff:(length-land),:], pos_ref_vector[takeoff:length-land,:], gp_input[takeoff:length-land,:], disturbance[takeoff:length-land,:]