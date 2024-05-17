from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
#from geometry_msgs.msg import TransformStamped
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


topic_name = '/drone/combined_data' # Add more topics as needed
bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/combined_data' ## ROS bag
bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/plotter/circular_trajectory_slow_with_disturbance'
typestore = get_typestore(Stores.LATEST)

msg_text = Path("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/DynamicsData.msg").read_text()

# Plain dictionary to hold message definitions.
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
        print(item.topic, item.msgtype)
    for item, timestamp, rawdata in reader.messages():
        if item.topic == topic_name:
            msg = typestore.deserialize_cdr(rawdata, item.msgtype)
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
            thrust = -kx*diff_pos -kv*diff_vel - g*m + m * acc_ref
            thrust = thrust  + g * m
            acc_cmd = thrust/m
            acc_diff = acc - acc_cmd
            disturbance.append(acc_diff)

            #print(msg.header.frame_id)

pos_arr = np.array(pos_arr)
vel_arr = np.array(vel_arr)
input = np.hstack((pos_arr, vel_arr))
disturbance = np.array(disturbance)
print(pos.shape)
print(vel.shape)
print(input.shape)
print(disturbance.shape)
#np.save("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/input_to_gp", input)
np.save("/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/disturbance", disturbance)
# np.save("pos",pos)
# np.save("velocity",vel)
