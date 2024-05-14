from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='trajectory_driver',  
            executable='data_node',  
            name='data',
            output='screen',
            parameters=[
                {'radius': 0.2},
                {'height': -0.4},
                {'center_x': 0.0},
                {'center_y': 0.0},
                {'angular_vel': 1.0},
                {'g': 9.81},
                {'kv': 7.4},
                {'kx': 14},
                {'m': 0.681},
            ]
        ),
        Node(
            package='trajectory_driver', 
            executable='message_node',  
            name='message_node',
            output='screen',
            parameters=[
                {'radius': 0.2},
                {'height': -0.4},
                {'center_x': 0.0},
                {'center_y': 0.0},
                {'angular_vel': 1.0},
            ]
        ),
    ])
