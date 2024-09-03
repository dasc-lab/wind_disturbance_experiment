from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
all_data_path = 'datasets/all_data/'
# [takeoff, landing] = [500, 800] 

input  = 
wind_disturbance = 
np.save(all_data_path+'input.npy',input)
np.save(all_data_path+'disturbance.npy',wind_disturbance)