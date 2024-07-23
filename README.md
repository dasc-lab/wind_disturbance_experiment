# Wind Disturbance Experiment Workflow
## Preliminaries
1. Use rope to do experiments especially for new tanh parameters
2. SSH xavier and `cd albus/rover_(tab complete)` and `docker compose up -d`
3. `ros2 launch all_launch px4.launch.py` on xavier
## On Xavier
1. Use tmux to divide the terminal window to three panes
2. First pane: `vim colcon_ws/src/trajectory_driver/trajectory_driver/circle_simple.py` change parameters such as radius, center coordinate, angular velocity and height
3. Second pane: `vim colcon_ws/src/trajectory_driver/launch/circle_simple.launch.py`. in the first code block after `-o`, change desired file name to the same parameters as previous step. r = radius, w = angular velocity, c = center, h = height, kxv = gains, delete `fanoff` if needed.
4. Similar steps for figure8
4. To record circle data, `ros2 launch trajectory_driver circle_simple.launch.py`
5. To record figure8 data, `ros2 launch trajectory_driver figureeight.launch.py` 


# Gaussian Process and Simulation
See GP/gp_advanced/README.md for detailed walkthrough.
