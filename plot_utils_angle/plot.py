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
plot_home = "media/"

circle_path = recorded_data_path + 'circle_data/'

figure8_path = recorded_data_path +'figure8_data/'

bag_path = circle_path + '08_07_2024_cir_traj_r0.4_w1.0_c0.0_h0.5_kx7_kv4_fanon_clipped'


# def calculate_reward(pos_ref, pos, vel_ref, vel, pos_factor = 1.0, vel_factor = 0.1):
#     reward = pos_factor * np.sum(np.fromiter((x**2 for x in (pos-pos_ref)),dtype=float)) + vel_factor * np.sum(np.fromiter((v**2 for v in (vel - vel_ref)),dtype =float))
#     return reward
def calculate_reward(pos_ref, pos, vel_ref, vel, pos_factor = 1.0, vel_factor = 0.1):
    reward = pos_factor * sum((x**2 for x in (pos-pos_ref))) + vel_factor * sum((v**2 for v in (vel - vel_ref)))
    return reward

def plot_trajectory_reward(bag_path, takeoff, land):
    
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
                # import pdb
                # pdb.set_trace()
                msg = typestore.deserialize_cdr(rawdata, item.msgtype)
                pos_vector.append(msg.pos)
                vel_vector.append(msg.vel)
                acc_vector.append(msg.acc)
                # print(f"{msg.kx}, {msg.kv}, {msg.quaternion}, {msg.angles}")
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

# optimized
pos_vector, pos_ref_vector, reward_arr, total_reward = plot_trajectory_reward(bag_path,0,0) # circle
# pos_vector, pos_ref_vector, reward_arr, total_reward = plot_trajectory_reward(bag_path,800,1400)
pos_vector = np.array(pos_vector)
reward_arr = np.array(reward_arr)

fig1, ax1  =plt.subplots()
ax1.plot(reward_arr, 'g', label='Optimized')

ax1.set_xlabel('horizon')
ax1.set_ylabel('Cost')
ax1.legend()
fig1.savefig(plot_home+"circle_cost.png")
fig1.savefig(plot_home+"circle_cost.eps")

fig2 = plt.figure() #(figsize=plt.figaspect(0.5))
ax2 = fig2.add_subplot(111, projection='3d')
ns = 10
width = 2
ax2.plot(pos_vector[1::ns,0],pos_vector[1::ns,1],pos_vector[1::ns,2], c = 'g', linewidth=width, label='Optimized')
ax2.plot(pos_ref_vector[1::ns,0],pos_ref_vector[1::ns,1],pos_ref_vector[1::ns,2], c='r',linewidth=width, linestyle='dotted', label='Reference')

ax2.set_zlim(0,1)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.legend()
fig2.savefig(plot_home+"circle_paths.png")
fig2.savefig(plot_home+"circle_paths.eps")
plt.show()


# plt.figure()
# plt.plot(reward_arr)
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pos_vector[:,0],pos_vector[:,1],pos_vector[:,2], c = 'r')
# ax.scatter(pos_ref_vector[:,0],pos_ref_vector[:,1],pos_ref_vector[:,2], c = 'b')
# ax.set_zlim(0,1)
# plt.show()