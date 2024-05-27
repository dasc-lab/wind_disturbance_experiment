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
from plot_trajectory_ref_circle import bag_path, cutoff, threshold
print("bag path is: ", bag_path)
print("cutoff = ", cutoff)
print("threshold = ", threshold)


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
acc = []
acc_ref = 0
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
            #thrust = thrust  + g * m
            acc_cmd = thrust/m
            acc_diff = acc - acc_cmd
            disturbance.append(acc_diff)

################## Preparing new data points ##################
new_pos_arr = np.array(pos_arr)
new_vel_arr = np.array(vel_arr)
new_input = np.hstack((new_pos_arr, new_vel_arr))
new_disturbance = np.array(disturbance)
print("new input shape = ", new_input.shape)
print("new_disturbance shape = ", new_disturbance.shape)
new_input = new_input[threshold:cutoff,:]
new_disturbance = new_disturbance[threshold:cutoff,:]
print("new input shape = ", new_input.shape)
print("new_disturbance shape = ", new_disturbance.shape)

################## loading previous datapoints ##################
input_file_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/input.npy'
disturbance_file_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/disturbance.npy'

################## Prepare input ##################

file_path = input_file_path
try:
    curr_input = np.load(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}. Creating a new blank file.")
    empty_data = np.empty((0, 6))
    np.save(file_path, empty_data)
    curr_input = np.load(file_path)
    assert curr_input.shape == (0,6)
print("current input shape = ", curr_input.shape)
if np.array_equal(new_input, curr_input) == False:
    print("concatenating input")
    input = np.vstack((curr_input, new_input))
    assert input.shape == (curr_input.shape[0] + new_input.shape[0], curr_input.shape[1])
else:
    input = curr_input
    print("new and current inputs are the same, not concatenating")
assert curr_input.shape[1] == new_input.shape[1] == input.shape[1] ==6

################## Prepare Output ##################

file_path = disturbance_file_path
try:
    curr_disturbance = np.load(file_path, allow_pickle=True)
except FileNotFoundError:
    print(f"File not found: {file_path}. Creating a new blank file.")
    empty_data = np.empty((0, 3))
    np.save(file_path, empty_data)
    
    curr_disturbance = np.load(file_path)
    assert curr_disturbance.shape == (0,3)
disturbance = np.vstack((curr_disturbance, new_disturbance))
if np.array_equal(new_disturbance, curr_disturbance) is False:
    print("concatenating disturbance")
    disturbance = np.vstack((curr_disturbance,new_disturbance))
    assert disturbance.shape == (curr_disturbance.shape[0] + new_disturbance.shape[0], curr_disturbance.shape[1])
else:
    disturbance = curr_disturbance
    print("new and current disturbances are the same, not concatenating")
assert new_disturbance.shape[1] == curr_disturbance.shape[1]== disturbance.shape[1] == 3, "shapes are different"

################## Save data points ##################

np.save(input_file_path, input)
np.save(disturbance_file_path, disturbance)
