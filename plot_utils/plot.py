import os
current_file_path = os.path.abspath(__file__)
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
recorded_data_path = 'recorded_data/'

circle_path = recorded_data_path + 'circle_data/'

figure8_path = recorded_data_path +'figure8_data/'

bag_path = circle_path + '28_07_2024_take1_cir_traj_r0.4_w1.0_c0.00_h0.5_kxv19_04_9_30_tank_0_31_fanon_clipped_new'

# def calculate_reward(pos_ref, pos, vel_ref, vel, pos_factor = 1.0, vel_factor = 0.1):
#     reward = pos_factor * np.sum(np.fromiter((x**2 for x in (pos-pos_ref)),dtype=float)) + vel_factor * np.sum(np.fromiter((v**2 for v in (vel - vel_ref)),dtype =float))
#     return reward
def calculate_reward(pos_ref, pos, vel_ref, vel, pos_factor = 1.0, vel_factor = 0.1):
    reward = pos_factor * sum((x**2 for x in (pos-pos_ref))) + vel_factor * sum((v**2 for v in (vel - vel_ref)))
    return reward

def plot_trajectory_reward(bag_path, takeoff, land):
    # x_data = []
    # y_data = []
    # z_data = x_ref = y_ref = z_ref = 
    pos_vector = []
    pos_ref_vector = []
    vel_vector = []
    vel_ref_vector = []
    acc_vector = []
    acc_ref = []
    reward_arr = []
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
            
                pos_ref_vector.append(msg.pos_ref)
                vel_ref_vector.append(msg.vel_ref)
                reward_t = calculate_reward(msg.pos_ref, msg.pos, msg.vel_ref, msg.vel)
                reward_arr.append(reward_t)
        
        pos_vector = np.array(pos_vector)
        pos_ref_vector = np.array(pos_ref_vector)
        reward_arr = np.array(reward_arr)
        length = pos_vector.shape[0]

        pos_vector[:,2] = -pos_vector[:,2]
        pos_ref_vector[:,2] = -pos_ref_vector[:,2]
    return pos_vector[takeoff:length-land,:], pos_ref_vector[takeoff:length-land,:], reward_arr[takeoff:length-land], np.sum(reward_arr[takeoff:length-land])
pos_vector, pos_ref_vector, reward_arr, total_reward = plot_trajectory_reward(bag_path,500,900)
pos_vector = np.array(pos_vector)
reward_arr = np.array(reward_arr)
plt.figure()
plt.plot(reward_arr)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos_vector[:,0],pos_vector[:,1],pos_vector[:,2], c = 'r')
ax.scatter(pos_ref_vector[:,0],pos_ref_vector[:,1],pos_ref_vector[:,2], c = 'b')
ax.set_zlim(0,1)
plt.show()