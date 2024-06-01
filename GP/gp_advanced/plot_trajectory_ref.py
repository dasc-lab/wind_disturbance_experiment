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



save = True
# Open the ROS2 bag file
#bag = rosbag.Bag('drone_trajectory.bag')
radius = 0.2
height = 0.4
angular_vel = 1.0 #rad/s
center_x = 0
center_y = 0

home_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/recorded_data/'

########################################################################
######################## Circle Paths ##################################
########################################################################
##### NOTE: The 'threshold' and 'cutoff' for each dataset are calculated and displayed beneath the bag_path of the dataset ######
##### NOTE: Replace the 'threshold' and 'cutoff' variables in this file with the value beneath each bag_path. Please do not uncomment the values #####
home_path = home_path + 'circle_data/'
#bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/data_prep/cir_traj_r0.4_w2_c0.40_h0.4_fanhigh'
#(1200, len(x_data)-700)
#bag_path = home_path + 'cir_traj_r0.3_w1.5_c00.4_h0.4_fanhigh'
#(1600, len(x_data)-1200)
#bag_path = home_path + 'cir_traj_r0.3_w1.5_c0.40_h0.4_fanhigh'
#(1400, len(x_data)-900)
#bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/recorded_data/cir_traj_r0.4_w2.5_c0.60_h0.4_fanhigh'
#(780, len(x_data) -1600)
#bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/recorded_data/cir_traj_r0.4_w3_c0.80_h0.4_fanhigh'
#(1000, len(x_data)-1000)
#bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/recorded_data/cir_traj_r0.4_w3_c10_h0.4_fanhigh'
#(800, len(x_data)-800)
#bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/recorded_data/eight_traj_r0.4_w2_c0.40_h0.4_fanhigh'
#(1000, len(x_data) - 2500)

########################################################################
######################## Figure Eight Paths ############################
########################################################################
home_path = home_path.replace('circle_data', 'eight_data')

bag_path = home_path + 'eight_traj_r0.2_w1.5_c0.80_h0.4_fanhigh'
#(threshold, cutoff) = (1200, len(x_data)-3500)
#bag_path = home_path + 'eight_traj_r0.2_w2_c1.20_h0.4_fanhigh'
#(threshold, cutoff) = (200, len(x_data)-800)
#bag_path = home_path + 'eight_traj_r0.2_w2.5_c1.20_h0.4_fanhigh'
#(threshold, cutoff) = (600, len(x_data)-600)
#bag_path = home_path + 'eight_traj_r0.4_w1.5_c1.20_h0.4_fanhigh'
#(threshold, cutoff) = (100, len(x_data)-100)
#bag_path = home_path + 'eight_traj_r0.4_w1.5_c10_h0.4_fanhigh'
#(threshold, cutoff) = (600, len(x_data)-800)
#bag_path = home_path + 'eight_traj_r0.4_w2_c0.40_h0.4_fanhigh'
#(threshold, cutoff) = (1000, len(x_data)-3000)
#bag_path = home_path + 'eight_traj_r0.4_w2_c10_h0.4_fanhigh'
#(threshold, cutoff) = (200, len(x_data)-600)
#bag_path = home_path + 'eight_traj_r0.4_w2.5_c10_h0.4_fanhigh'
#(threshold, cutoff) = (100, len(x_data)-600)
#bag_path = home_path + 'eight_traj_r0.6_w1.5_c10_h0.4_fanhigh'
#(threshold, cutoff) = (100, len(x_data)-100)



# Define the topics we want to extract data from
topic_name = '/drone/combined_data' # Add more topics as needed
png_name = bag_path.split('/')[-1]+'_trajectory'



#bag_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/data_prep/eight_traj_r0.4_w2_c0.40_h0.4_fanhigh'
typestore = get_typestore(Stores.LATEST)

msg_text = Path('/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/DynamicsData.msg').read_text()

add_types = {}

# Add definitions from one msg file to the dict.
add_types.update(get_types_from_msg(msg_text, 'foresee_msgs/msg/DynamicsData'))
typestore.register(add_types)


# Initialize empty lists to store concatenated x, y, and z data
x_data = []
y_data = []
z_data = []
timestamps = []

x_ideal = []
y_ideal = []
z_ideal = []

with Reader(bag_path) as reader:
    for item in reader.connections:
        print(item.topic, item.msgtype)
    for item, timestamp, rawdata in reader.messages():
        if item.topic == topic_name:
            msg = typestore.deserialize_cdr(rawdata, item.msgtype)
            x_data.append(msg.pos[0])
            y_data.append(msg.pos[1])
            z_data.append(-msg.pos[2])
            
            pos_ref = msg.pos_ref
            vel_ref = msg.vel_ref
            acc_ref = msg.acc_ref
            x_ideal.append(pos_ref[0])
            y_ideal.append(pos_ref[1])
            z_ideal.append(-pos_ref[2])
           

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
assert len(x_data) == len(y_data) == len(z_data), "Lengths of the lists are not the same."
cutoff = len(x_data) - 3500
threshold = 1200


print("cutoff, threshold = ", cutoff, threshold)
print("data set size = ",  cutoff - threshold)
# indices = np.arange(1, len(x_data) + 1)
# print(len(indices))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(x_data[threshold:cutoff])
y = np.array(y_data[threshold:cutoff])
z = np.array(z_data[threshold:cutoff])
#print("size of x data is: ",x.shape)
x_t = x_ideal = np.array(x_ideal[threshold:cutoff])
y_t = y_ideal = np.array(y_ideal[threshold:cutoff])
z_t = z_ideal = np.array(z_ideal[threshold:cutoff])

# Scatter plot
ax.set_zlim(0,1)
ax.scatter(x, y, z, c= 'r')
ax.scatter(x_ideal, y_ideal, z_ideal, c = 'b')
#plt.savefig("trajectory.png")
plt.show()

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
png_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/trajectory_sinusoid_plots/' + png_name +'.png'
plt.suptitle(png_name)
plt.savefig(png_path)
plt.show()
#input = input("save trajectory to csv? y/n")
input = 'n'
if input == 'y':
    save = True
elif input == 'n':
    save = False
else:
    print("invalid input")
    save = False
## save to csv
if save :
    rows = zip(x_data, y_data, z_data)
    filename = "drone_trajectory.csv"
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(rows)
        
    print(f"Three lists saved to {filename}.")
# data_prep_path = os.path('/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/')
# sys.path.append(data_prep_path)
# import data_prep.gp_data