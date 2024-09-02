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

bag_path = circle_path + '28_07_2024_take1_cir_traj_r0.4_w1.0_c0.00_h0.5_kxv7_00_4_00_tank_0_31_fanoff_clipped_new'

def calculate_reward(pos_ref, pos, vel_ref, vel, pos_factor = 1.0, vel_factor = 0.1):
    reward = pos_factor * np.sum(np.fromiter((x**2 for x in (pos-pos_ref)),dtype=float)) + vel_factor * np.sum(np.fromiter((v**2 for v in (vel - vel_ref)),dtype =float))
    return reward

def plot_trajectory_reward(bag_path, takeoff, land):
    x_data = y_data = z_data = x_ref = y_ref = z_ref = pos_vector = vel_vector = acc_vector = reward_arr = []
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
                # pos_vector.append(msg.pos)
                # vel_vector.append(msg.vel)
                # acc_vector.append(msg.acc)
                x_data.append(msg.pos[0])
                y_data.append(msg.pos[1])
                z_data.append(-msg.pos[2])
                
                pos_ref = msg.pos_ref
                vel_ref = msg.vel_ref
                acc_ref = msg.acc_ref
                x_ref.append(pos_ref[0])
                y_ref.append(pos_ref[1])
                z_ref.append(-pos_ref[2])
                reward_t = calculate_reward(pos_ref, msg.pos,vel_ref, msg.vel)
                reward_arr.append(reward_t)
            pos_vector = np.column_stack((x_data,y_data,z_data))
            pos_ref_vector = np.column_stack((x_ref,y_ref,z_ref))
            reward_arr = np.array(reward_arr)
    return pos_vector, pos_ref_vector, reward_arr
pos_vector, pos_ref_vector, reward_t = plot_trajectory_reward(bag_path,500,500)
pos_vector = np.array(pos_vector)
reward_t = np.array(reward_t)
print(pos_vector.shape)
print(reward_t.shape)