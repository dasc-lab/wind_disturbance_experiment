import numpy as np
import numpy.random as npr

input_file_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/input.npy'
disturbance_file_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/disturbance.npy'

initial_input = np.load(input_file_path)
initial_disturbance = np.load(disturbance_file_path)
assert initial_input.shape[1] == 6 and initial_disturbance.shape[1] == 3 and initial_input.shape[0] == initial_disturbance.shape[0]
print("initial input shape: ", initial_input.shape)
print("initial disturbance shape: ", initial_disturbance.shape)
###### filter data so one out of three datapoints is saved ######
filtered_input = initial_input[::3]
filtered_disturbance = initial_disturbance[::3]
assert filtered_input.shape == (initial_input.shape[0]/3 + 1, initial_input.shape[1]) or filtered_disturbance.shape == (initial_input.shape[0]/3, initial_input.shape[1])
print("filtered input shape:", filtered_input.shape)
print("filtered disturbance shape", filtered_disturbance.shape)
###### Prepare the train, test, and entire data set ######
fullset_input = filtered_input
fullset_disturbance = filtered_disturbance
fullset_input_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/fullset_input.npy'
fullset_disturbance_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/fullset_disturbance.npy'
np.save(fullset_input_path, fullset_input)
np.save(fullset_disturbance_path,fullset_disturbance)


###### Prepare the train data set ######
training_set_size = int(fullset_input.shape[0] * 0.8)
training_indices = npr.choice(fullset_input.shape[0],training_set_size, replace=False)
assert max(training_indices) < fullset_input.shape[0], "sampling error: indices need to be in bound"
training_set_input = fullset_input[training_indices]
training_set_disturbance = fullset_disturbance[training_indices]
training_set_input_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/training_input.npy'
training_set_disturbance_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/training_disturbance.npy'
np.save(training_set_input_path, training_set_input)
np.save(training_set_disturbance_path, training_set_disturbance)
print("training input shape:", training_set_input.shape)
print("training disturbance shape: ", training_set_disturbance.shape)

###### Prepare the test data set ######
test_set_size = fullset_input.shape[0] - training_set_size
test_set_input = np.delete(fullset_input, training_indices)
test_set_disturbance = np.delete(fullset_disturbance, training_indices)
test_set_input_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/test_input.npy'
test_set_disturbance_path = '/Users/albusfang/Coding Projects/gp_ws/Gaussian Process/GP/gp_advanced/test_disturbance.npy'
np.save(test_set_input_path, test_set_input)
np.save(test_set_disturbance_path, test_set_disturbance)
print("test input shape:", test_set_input.shape)
print("test disturbance shape: ", test_set_disturbance.shape)
