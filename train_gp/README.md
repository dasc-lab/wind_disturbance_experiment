# Training and Testing Gaussian Process Models
1. Run `npy_files.py` to truncate the takeoff and landing data from every trajectory by entering a value for each of `takeoff` and `landing` when prompted and concatenate the results together and save them as `disturbance.npy` and `input.npy` in `/dataset/all_data/`.
2. Use `save_data.py` to create training and test set randomly and save the datasets in the `training/` and `testing/` folders respectively.
3. Train Sparse GP Models with `gp_sparse.py`.
4. Run the model on test data with `load_gp.py`
