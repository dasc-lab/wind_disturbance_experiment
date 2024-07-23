# Wind Disturbance Experiment Workflow
## Preliminaries
1. Use rope to do experiments especially for new tanh parameters
2. SSH xavier and `cd albus/rover_(tab complete)` and `docker compose up -d`
3. `ros2 launch all_launch px4.launch.py` on xavier
## On Xavier
3. To record circle data, `ros2 launch trajectory_driver circle_simple.launch.py`
4. To record figure8 data, `ros2 launch trajectory_driver figure8.launch.py`
5. `cd colcon_ws/src/`

# Gaussian Process and Simulation
See GP/gp_advanced/README.md for more info.
