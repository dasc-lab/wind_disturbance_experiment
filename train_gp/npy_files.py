import os
current_file_path = os.path.abspath(__file__)

from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import plot_helper, plot_trajectory
recorded_data_path = 'recorded_data/'

circle_path = recorded_data_path + 'circle_data/training/'

figure8_path = recorded_data_path +'figure8_data/training/'
npy_path = 'all_data/'

all_input_data = npy_path + 'input.npy'
all_disturbance_data = npy_path + 'disturbance.npy'

for file_name in os.listdir(circle_path):
    file_path = os.path.join(circle_path, file_name)
    
    if os.path.isdir(file_path):
        print(f"Processing data: {file_name}")
    while (input("Is takeoff and landing data truncated?[y/n]") != 'y'):
        takeoff = input("Enter takeoff threshold: ")
        landing = input("Enter landing cutoff: ")
        pos_vector, pos_ref_vector= plot_trajectory(file_path,takeoff,landing)
        pos_vector = np.array(pos_vector)
        plot_helper(pos_vector,pos_ref_vector)
    

    try:
        curr_input = np.load(all_input_data)
    except FileNotFoundError:
        print(f"File not found: {all_input_data}. Creating a new blank file.")
        empty_data = np.empty((0, 6))
        np.save(all_input_data, empty_data)
        curr_input = np.load(all_input_data)
        assert curr_input.shape == (0,6)

    try:
        curr_disturbance = np.load(all_disturbance_data, allow_pickle=True)
    except FileNotFoundError:
        print(f"File not found: {all_disturbance_data}. Creating a new blank file.")
        empty_data = np.empty((0, 3))
        np.save(all_disturbance_data, empty_data)
        
        curr_disturbance = np.load(all_disturbance_data)
        assert curr_disturbance.shape == (0,3)
    gp_input = np.vstack((curr_input, new_input))
    disturbance = np.vstack((curr_disturbance, new_disturbance))





for file_name in os.listdir(figure8_path):
    file_path = os.path.join(figure8_path, file_name)
    # Check if it is a file (not a directory)
    if os.path.isdir(file_path):
        print(f"Processing data: {file_name}")
    while (input("Is takeoff and landing data truncated?[y/n]") != 'y'):
        takeoff = input("Enter takeoff threshold: ")
        landing = input("Enter landing cutoff: ")
        pos_vector, pos_ref_vector= plot_trajectory(file_path,takeoff,landing)
        pos_vector = np.array(pos_vector)
        plot_helper(pos_vector,pos_ref_vector)
    
    gp_input = np.vstack((curr_input, new_input))
    disturbance = np.vstack((curr_disturbance, new_disturbance))
