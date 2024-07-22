##################################################################################################################################################################
############ Use this file to visualize the trajectory taken and the reference trajecotry.                                  #############################
############ Adjust the values of 'threshold' (lower bound) and 'cutoff' (upper bound) to crop off takeoff and landing data #############################
##################################################################################################################################################################
#ros2 bag record -o drone_trajectory /px4_1/fmu/out/
#ros2 bag info drone_trajectory.bag

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
#from geometry_msgs.msg import TransformStamped
from pathlib import Path
import numpy as np


import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt
def fft_filter(signal, sampling_rate):
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

    cutoff_freq = peak_frequency+3 #Hz
    filtered_signal = filtered_data = butter_lowpass_filter(signal, cutoff_freq, sampling_rate)
    return filtered_signal
def apply_fft_filter_to_columns(array, sampling_rate):
    filtered_array = np.zeros_like(array)
    for i in range(array.shape[1]):
        filtered_array[:, i] = fft_filter(array[:, i], sampling_rate)
    return filtered_array


save = True
# Open the ROS2 bag file
#bag = rosbag.Bag('drone_trajectory.bag')
# radius = 0.2
# height = 0.4
# angular_vel = 1.0 #rad/s
# center_x = 0
# center_y = 0
# kx = 14
# kv = 7.4
kx = 14.0
kv = 7.4
m = 0.681
g = 9.81


repo_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/'
home_path = repo_path + 'GP/gp_advanced/recorded_data/'
plot_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/trajectory_sinusoid_plots/'

########################################################################
######################## Circle Paths ##################################
########################################################################
##### NOTE: The 'threshold' and 'cutoff' for each dataset are calculated and displayed beneath the bag_path of the dataset ######
##### NOTE: Replace the 'threshold' and 'cutoff' variables in this file with the value beneath each bag_path. Please do not uncomment the values #####
home_path = home_path + 'circle_data/'

# bag_path = home_path + 'cir_traj_r0.2_w1.0_c00_h0.5_kxv74_clipped_new'
# (200,500)

# bag_path = home_path + 'cir_traj_r0.2_w1.0_c0.60_h0.5_kxv74_clipped_new'
# (500,500)

# bag_path = home_path + 'cir_traj_r0.2_w1.0_c0.80_h0.5_kxv74_clipped_new'
# (800, len-500)


# bag_path = home_path + 'cir_traj_r0.4_w1.0_c0.60_h0.5_kxv74_clipped_new'
# (500,500)

################## fanoff data ##################
# bag_path = home_path + 'cir_traj_r0.2_w1.0_c00_h0.5_kxv74_fanoff_clipped_new'
# (500, 2500)

########################################################################
######################## Figure Eight Paths ############################
########################################################################

home_path = home_path.replace('circle_data', 'eight_data')

bag_path = home_path + 'eight_traj_r0.4_w1_c00_h0.5_kxv74_clipped_new'
# (500,500)

# bag_path = home_path + 'eight_traj_r0.4_w1_c0.80_h0.5_kxv74_clipped_new'
# (1200, 1000)




# bag_path = home_path + 'eight_traj_r0.4_w1_c00_h0.5_kxv74_fanoff_clipped_new'
# (500,500)






# Define the topics we want to extract data from
topic_name = '/drone/combined_data' # Add more topics as needed
png_name = bag_path.split('/')[-1]+'_trajectory_acc_cmd_disturbance'



#bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/data_prep/eight_traj_r0.4_w2_c0.40_h0.4_fanhigh'
typestore = get_typestore(Stores.LATEST)

msg_text = Path(repo_path+'DynamicsData.msg').read_text()

add_types = {}

# Add definitions from one msg file to the dict.
add_types.update(get_types_from_msg(msg_text, 'foresee_msgs/msg/DynamicsData'))
typestore.register(add_types)


# Initialize empty lists to store concatenated x, y, and z data
pos_arr = []

vel_arr = []
acc = []
acc_ref = 0
disturbance = []
acc_cmd_arr = []
acc_arr = []

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
            # if np.linalg.norm(diff_pos) > 2.0:
            #     diff_pos = 2.0 * diff_pos/ np.linalg.norm(diff_pos)
            # if np.linalg.norm(diff_vel) > 5.0:
            #     diff_vel = 5.0 * diff_vel/ np.linalg.norm(diff_vel)
            thrust = -kx*diff_pos - kv*diff_vel + m * acc_ref #- g*m 
            #thrust = thrust  + g * m
            acc_cmd = thrust/m
            acc_diff = acc - acc_cmd
            disturbance.append(acc_diff)
            acc_cmd_arr.append(acc_cmd)
            acc_arr.append(acc)
# Iterate through the messages in the bag file for the current topic
# for topic, msg, t in bag.read_messages(topics=[topic_name]):
#     # Extract relevant data from the message
#     x_data.append(msg.x)
#     y_data.append(msg.y)
#     z_data.append(msg.z)
# bag.close()
# print("x max: ", max(x_data))
# print("x min: ", min(x_data))
# print("y max: ", max(y_data))
# print("y min: ", min(y_data))
# print("z max: ", max(z_data))
# print("z min: ", min(z_data))
# assert len(x_data) == len(y_data) == len(z_data), "Lengths of the lists are not the same."
acc_cmd_arr = np.array(acc_cmd_arr)
unfiltered_acc_arr = recorded_acc_arr = np.array(acc_arr)
recorded_acc_arr = apply_fft_filter_to_columns(recorded_acc_arr, sampling_rate=1000)
disturbance = recorded_acc_arr - acc_cmd_arr
cutoff = len(acc_cmd_arr) - 500
threshold = 500

print("cutoff, threshold = ", cutoff, threshold)
print("data set size = ",  cutoff - threshold)
# indices = np.arange(1, len(x_data) + 1)
# print(len(indices))
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
x = np.array(acc_cmd_arr[threshold:cutoff,0])
y = np.array(acc_cmd_arr[threshold:cutoff,1])
z = np.array(acc_cmd_arr[threshold:cutoff,2])


disturbance_x = np.array(disturbance[threshold:cutoff,0])
disturbance_y = np.array(disturbance[threshold:cutoff,1])
disturbance_z = np.array(disturbance[threshold:cutoff,2])

recorded_acc_arr_x = np.array(recorded_acc_arr[threshold:cutoff,0])
recorded_acc_arr_y = np.array(recorded_acc_arr[threshold:cutoff,1])
recorded_acc_arr_z = np.array(recorded_acc_arr[threshold:cutoff,2])

unfiltered_acc_arr_x = np.array(unfiltered_acc_arr[threshold:cutoff,0])
unfiltered_acc_arr_y = np.array(unfiltered_acc_arr[threshold:cutoff,1])
unfiltered_acc_arr_z = np.array(unfiltered_acc_arr[threshold:cutoff,2])

clip_val = 3
plt.figure()
plt.plot(range(unfiltered_acc_arr_x.size), unfiltered_acc_arr_x , 'r',label='unfiltered recorded acc')
plt.plot(range(recorded_acc_arr_x.size),  recorded_acc_arr_x, 'b',label='Acc after filter')
plt.ylim(-clip_val,clip_val)
plt.legend()
plt.show()

plt.figure()
plt.plot(range(unfiltered_acc_arr_y.size), unfiltered_acc_arr_y , 'r',label='unfiltered recorded acc')
plt.plot(range(recorded_acc_arr_y.size),  recorded_acc_arr_y, 'b',label='Acc after filter')
plt.ylim(-clip_val,clip_val)
plt.legend()
plt.show()

plt.figure()
plt.plot(range(unfiltered_acc_arr_z.size), unfiltered_acc_arr_z , 'r',label='unfiltered recorded acc')
plt.plot(range(recorded_acc_arr_z.size),  recorded_acc_arr_z, 'b',label='Acc after filter')
plt.ylim(-clip_val,clip_val)
plt.legend()
plt.show()
#print("size of x data is: ",x.shape)
# x_t = x_ideal = np.array(x_ideal[threshold:cutoff])
# y_t = y_ideal = np.array(y_ideal[threshold:cutoff])
# z_t = z_ideal = np.array(z_ideal[threshold:cutoff])

# Scatter plot
#ax.set_zlim(-2,3)
#ax.scatter(x, y, z, c= 'r')
#ax.scatter(x, y, c= 'b')
# ax.scatter(x_ideal, y_ideal, z_ideal, c = 'b')
#plt.savefig("trajectory.png")
#plt.show()

fig = plt.figure()
ax1 = plt.subplot2grid((3,1), (0,0) , rowspan=1)
ax1.plot(range(recorded_acc_arr_x.size), recorded_acc_arr_x , 'g',label='recorded_acc_x')
ax1.plot(range(disturbance_x.size), disturbance_x , 'r',label='disturbance x')
ax1.plot(range(x.size), x , 'b',label='Acc command x')
# ax1.plot(range(x_t.size), x_t.T, 'b',label='Reference trajectory')
ax1.legend(loc='upper right')
ax1.set_title("acc_cmd_x and disturbance_x")
ax1.set_ylabel("meters/second^2")
ax1 = plt.subplot2grid((3,1), (1,0) , rowspan=1)
ax1.plot(range(recorded_acc_arr_y.size), recorded_acc_arr_y , 'g',label='recorded_acc_y')
ax1.plot(range(disturbance_y.size), disturbance_y , 'r',label='disturbance y')
ax1.plot(range(y.size), y,'b',label='Acc command y')
# ax1.plot(range(y_t.size), y_t.T, 'b',label='Reference trajectory')
ax1.legend(loc='upper right')
ax1.set_title("acc_cmd_y and disturbance_y")
ax1.set_ylabel("meters/second^2")
ax1 = plt.subplot2grid((3,1), (2,0) , rowspan=1)
ax1.plot(range(recorded_acc_arr_z.size), recorded_acc_arr_z , 'g',label='recorded_acc_z')
ax1.plot(range(disturbance_z.size), disturbance_z , 'r',label='disturbance z')
ax1.plot(range(z.size), z,'b',label='Actual trajectory')
# ax1.plot(range(z_t.size), z_t.T, 'b',label='Reference trajectory')
ax1.legend(loc='upper right')
ax1.set_ylabel("meters/second^2")
ax1.set_title("acc_cmd_z and disturbance_z")
plt.subplots_adjust(hspace=0.5) 
png_path = plot_path + png_name +'.png'
plt.suptitle(png_name)
plt.savefig(png_path)
plt.show()

plt.figure()
plt.plot(range(recorded_acc_arr_x.size), recorded_acc_arr_x , 'g',label='recorded_acc_x')
plt.plot(range(disturbance_x.size), disturbance_x , 'r',label='disturbance x')
plt.plot(range(x.size), x , 'b',label='Acc command x')
plt.title("x axis acc_recorded, acc_cmd, and disturbance")
plt.legend()
plt.ylabel("acc")
plt.show()

plt.figure()
plt.plot(range(recorded_acc_arr_y.size), recorded_acc_arr_y , 'g',label='recorded_acc_y')
plt.plot(range(disturbance_y.size), disturbance_y , 'r',label='disturbance y')
plt.plot(range(y.size), y , 'b',label='Acc command y')
plt.title("y axis acc_recorded, acc_cmd, and disturbance")
plt.ylabel("acc")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(recorded_acc_arr_z.size), recorded_acc_arr_z , 'g',label='recorded_acc_z')
plt.plot(range(disturbance_z.size), disturbance_z , 'r',label='disturbance z')
plt.plot(range(z.size), z , 'b',label='Acc command z')
plt.title("z axis acc_recorded, acc_cmd, and disturbance")
plt.ylabel("acc")
plt.legend()
plt.show()
#input = input("save trajectory to csv? y/n")
# input = 'n'
# if input == 'y':
#     save = True
# elif input == 'n':
#     save = False
# else:
#     print("invalid input")
#     save = False
# ## save to csv
# if save :
#     rows = zip(x_data, y_data, z_data)
#     filename = "drone_trajectory.csv"
#     with open(filename, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerows(rows)
        
#     print(f"Three lists saved to {filename}.")
# data_prep_path = os.path('/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/')
# sys.path.append(data_prep_path)
# import data_prep.gp_data