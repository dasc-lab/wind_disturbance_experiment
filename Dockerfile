#FROM ghcr.io/nvidia/jax:jax
# FROM nvcr.io/nvidia/jax:23.08-py3
FROM nvcr.io/nvidia/jax:23.10-py3
RUN apt-get update
RUN apt-get install -y gedit
RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
RUN pip3 install PyQt5
RUN pip3 install matplotlib
# RUN pip3 install gpjax==0.8.2
# RUN pip3 uninstall gpjax -y

## NOTE: gpjax version dependent on JAX version. Should change depending on which jax nvidia image has been pulled up

# with nvcr.io/nvidia/jax:23.10-py3
RUN pip3 install gpjax==0.8.0 


ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libqt5gui5
RUN apt-get install -y texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra dvipng cm-super
RUN apt install -y vim
# RUN pip3 uninstall gpjax -y

# ROS installation

RUN apt install -y software-properties-common
RUN add-apt-repository universe
RUN apt update && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
RUN apt install -y ros-humble-ros-base ros-dev-tools
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN apt-get install -y libboost-all-dev ros-humble-diagnostic-updater
RUN apt install -y tmux iputils-ping

RUN echo "alias gpsetup='source /home/wind_disturbance_experiment/GP/gp_advanced/FORESEE/setup.sh'" >> ~/.bashrc
# RUN export PYTHONPATH=$PYTHONPATH:/home/wind_disturbance_experiment/GP/gp_advanced/FORESEE/GPJax
RUN echo "alias sim='cd /home/wind_disturbance_experiment/GP/gp_advanced/simulation'" >> ~/.bashrc