# Wind Disturbance Experiment Workflow
## Data Recording
### Preliminaries
1. Use rope to do experiments especially for new tanh parameters
2. SSH xavier and `cd albus/rover_(tab complete)` and `docker compose up -d`
3. `ros2 launch all_launch px4.launch.py` on xavier
### On Xavier
1. Use tmux to divide the terminal window to three panes
2. First pane: `vim colcon_ws/src/trajectory_driver/trajectory_driver/circle_simple.py` change parameters such as radius, center coordinate, angular velocity and height
3. Second pane: `vim colcon_ws/src/trajectory_driver/launch/circle_simple.launch.py`. In the first code block after `-o`, change desired file name to the same parameters as previous step. `r` = radius, `w` = angular velocity, `c` = center, `h` = height, `kxv` = gains kx and kv, delete `fanoff` if needed. **To make drone closer to the fan, increase x value**.
4. Similar steps for figure8
5. To record circle data, `ros2 launch trajectory_driver circle_simple.launch.py`
6. To record figure8 data, `ros2 launch trajectory_driver figureeight.launch.py` 
7. Outside of docker, push changes to github with `git push origin foresee`. Then copy and paste data into recorded data folder.
# Gaussian Process and Simulation
See `GP/gp_advanced/README.md` for detailed walkthrough.
