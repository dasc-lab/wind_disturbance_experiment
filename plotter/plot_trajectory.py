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

save = True
# Open the ROS2 bag file
#bag = rosbag.Bag('drone_trajectory.bag')

# Define the topics you want to extract data from
topic_name = '/vicon/px4_1/px4_1' # Add more topics as needed
bag_path = '/rosbag'
bag_path = '/Users/albusfang/Coding Projects/Gaussian Process/trajectories/trajectory_circular'
typestore = get_typestore(Stores.LATEST)


# Initialize empty lists to store concatenated x, y, and z data
x_data = []
y_data = []
z_data = []
with Reader(bag_path) as reader:
    for item in reader.connections:
        print(item.topic, item.msgtype)
    for item, timestamp, rawdata in reader.messages():
        if item.topic == topic_name:
            msg = typestore.deserialize_cdr(rawdata, item.msgtype)
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
# Scatter plot
ax.set_zlim(0,0.5)
ax.scatter(x, y, z, c= 'b')
plt.savefig("trajectory.png")
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



