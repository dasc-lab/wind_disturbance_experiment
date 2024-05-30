# Gaussian Data and Model for Wind Experiements

## Reading recorded data, visualizing the trajectories, and truncating takeoff and landing data
Use `plot_trejectory_ref.py` to visualize the trajectories and its 3D components. Use `bag_path` variable to select different trajectories to be processed. Adjust variables `cutoff` and `threshold` to remove takeoff and landing data, where `threshold` removes takeoff data towards the beginning of the recording and `cutoff` removes landing data from the recording. Please ensure no takeoff and landing data is included as this can mess up Gaussian Process training. Other files such as `plot_acc.py` may import `cutoff` and `threshold`. 

## Visualize Acceleration and Applying Low-Pass Filter
Use `fft.py` to analyze the signal in frequency domain and visualize the signal pre and post filtering.
Use `plot_acc.py` to plot out filtered `acc_x`, `acc_y`, and `acc_z`. Use `fft.py` if you want to analyze the accelerations individually.
**Note**
`fft_filter(signal, sampling_rate = 5000)` in `plot_acc.py` and `fft.py` calculates the _cutoff frequency_ by adding a hard-coded value to the peak frequency, which may yield varing results. **A more robust way of calculating the cutoff frequency may be desirable**.

# Stitching Dataset and Generate Train and Test Dataset


To train GP on all collected datasets with various speeds and coordinates, one would need to crop off the takeoff and landing data from each data set and stitch the cropped datasets together. Run `gp_data_prep.py` to first filter the disturbance and then concatenate the data points together. If the new data set is the same as the existing dataset, the new data won't be concatenated.

## GP Training and Plotting

Use `gp.py` to train gp model for all 3 dimensions of the filtered disturbance with respect to all six inputs. The script saves its training dataset, test dataset, and the whole data set.
