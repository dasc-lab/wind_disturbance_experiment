#ros2 bag record -o drone_trajectory /px4_1/fmu/out/
#ros2 bag info drone_trajectory.bag
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
#from geometry_msgs.msg import TransformStamped
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
def draw_circle_3d(radius, height):
    

    # Parameters for the circle
    theta = np.linspace(0, 2*np.pi, 10000)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, height)
    return x,y,z
def draw_figureeight_3d(radius, height, length=10000):
    

    # Parameters for the circle
    theta = np.linspace(0, 2*np.pi, length)
    y = radius * np.sin(theta)
    x = radius * np.sin(theta) * np.cos(theta)
    z = np.full_like(theta, height)
    return x,y,z
def generate_coordinates(timestamps, radius = 0.2 , angular=1, height=0.4):
    timestamps = np.array([timestamps])
    current_time = timestamps[0]
    print(timestamps)

    indices = timestamps
    
    theta = angular*(indices)
    x = radius * np.sin(theta) * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, height)
    return x, y, z
save = True
# Open the ROS2 bag file
#bag = rosbag.Bag('drone_trajectory.bag')
radius = 0.2
height = 0.4
angular_vel = 1.0 #rad/s
center_x = 0
center_y = 0
# Define the topics you want to extract data from
topic_name = '/vicon/px4_1/px4_1' # Add more topics as needed
bag_path = '/Users/albusfang/Coding Projects/Gaussian Process/trajectories/trajectory_figureeight'
typestore = get_typestore(Stores.LATEST)


# Initialize empty lists to store concatenated x, y, and z data
x_data = []
y_data = []
z_data = []
timestamps = []

x_ideal = []
y_ideal = []
z_ideal = []
first_second = None
with Reader(bag_path) as reader:
    for item in reader.connections:
        print(item.topic, item.msgtype)
    for item, timestamp, rawdata in reader.messages():
        if item.topic == topic_name:
            msg = typestore.deserialize_cdr(rawdata, item.msgtype)
            
            # print("nano: ",  msg.header.stamp.nanosec/1e9)
            if first_second is not None: 
                print("seconds: ",msg.header.stamp.sec)
                print("first second: ", first_second)
                timestamp = msg.header.stamp.sec-first_second + msg.header.stamp.nanosec/10**9
            else:
                first_second = msg.header.stamp.sec
                timestamp = 0
            timestamps.append(timestamp)
            # x_ideal.append(radius * np.cos(angular_vel*(timestamp - timestamps[0])) + center_x)
            # y_ideal.append(radius * np.sin(angular_vel*(timestamp - timestamps[0]))*np.cos(angular_vel*(timestamp - timestamps[0])) + center_y)
            # z_ideal.append(height)
            trans = msg.transform.translation
            x_data.append(trans.x)
            y_data.append(trans.y)
            z_data.append(trans.z)
    
            print(msg.header.frame_id)
# Iterate through the messages in the bag file for the current topic
# for topic, msg, t in bag.read_messages(topics=[topic_name]):
#     # Extract relevant data from the message
#     x_data.append(msg.x)
#     y_data.append(msg.y)
#     z_data.append(msg.z)
# bag.close()
print("x max: ", max(x_data))
print("x min: ", min(x_data))
print("y max: ", max(y_data))
print("y min: ", min(y_data))
print("z max: ", max(z_data))
print("z min: ", min(z_data))
assert len(x_data) == len(y_data) == len(z_data), "Lengths of the lists are not the same."

# indices = np.arange(1, len(x_data) + 1)
# print(len(indices))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(x_data)
y = np.array(y_data)
z = np.array(z_data)

x_ideal,y_ideal, z_ideal = draw_figureeight_3d(radius, height)
x_t, y_t, z_t = generate_coordinates(timestamps, radius, angular_vel, height)
# Scatter plot
ax.set_zlim(0,0.6)
ax.scatter(x, y, z, c= 'r')
ax.scatter(x_ideal, y_ideal, z_ideal, c = 'b')
plt.savefig("trajectory_figureeight.png")
plt.show()
 


fig = plt.figure()
ax1 = plt.subplot2grid((3,1), (0,0) , rowspan=1)
ax1.plot(range(x.size), x, 'r')
ax1.plot(range(x_t.size), x_t.T, 'b')
ax1.set_title("x")
ax1.set_ylabel("meters")
ax1 = plt.subplot2grid((3,1), (1,0) , rowspan=1)
ax1.plot(range(y.size), y,'r')
ax1.plot(range(y_t.size), y_t.T, 'b')
ax1.set_title("y")
ax1.set_ylabel("meters")
ax1 = plt.subplot2grid((3,1), (2,0) , rowspan=1)
ax1.plot(range(z.size), z,'r')
ax1.plot(range(z_t.size), z_t.T, 'b')
ax1.set_ylabel("meters")
ax1.set_title("z")
plt.savefig("trajectory_individual_figureeight.png")
plt.show()

input = input("save trajectory to csv? y/n")
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



