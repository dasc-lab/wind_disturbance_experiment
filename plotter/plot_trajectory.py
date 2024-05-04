#ros2 bag record -o drone_trajectory /px4_1/fmu/out/
#ros2 bag info drone_trajectory.bag
import rosbag
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

save = True
# Open the ROS2 bag file
bag = rosbag.Bag('drone_trajectory.bag')

# Define the topics you want to extract data from
topic_name = '/px4_1/fmu/out/' # Add more topics as needed

# Initialize empty lists to store concatenated x, y, and z data
x_data = []
y_data = []
z_data = []

# Iterate through the messages in the bag file for the current topic
for topic, msg, t in bag.read_messages(topics=[topic_name]):
    # Extract relevant data from the message
    x_data.append(msg.x)
    y_data.append(msg.y)
    z_data.append(msg.z)

bag.close()

assert len(x_data) == len(y_data) == len(z_data), "Lengths of the lists are not the same."

# indices = np.arange(1, len(x_data) + 1)
# print(len(indices))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(x_data)
y = np.array(y_data)
z = np.array(z_data)
# Scatter plot
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



