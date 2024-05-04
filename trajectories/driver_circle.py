import rclpy
from rclpy.node import Node
from std_msgs.msg import *
import numpy as np
from geometry_msgs.msg import TransformStamped
from px4_msgs.msg import TrajectorySetpoint
from std_msgs.msg import String

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class driveCircle(Node):
    def __init__(self):
		## initializing the camera
        super().__init__('driveCircle')
		#self.output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30, (640,480))
        self.radius = 2.0
        self.height = 1.5
        self.num_points = 50
        self.publisher_ = self.create_publisher(TrajectorySetpoint, '/px4_1/fmu/in/trajectory_setpoint', 10)
        
        self.index = 0
        self.coordinate = None
        self.quat = None
        self.world_coordinate = None
        self.t = 0
        self.dt = 0.05

        ################## set up Subscription ##################
        self.waypoints = None
        self.robot_subscription = self.create_subscription(
		    TransformStamped,
		    #'/vicon/kepler/kepler',
		    #'/vicon/kepler/kepler',
		    '/vicon/px4_1/px4_1',
		    self.listener_callback_drone,
		    10)
        self.generate_trajectory()
        self.timer = self.create_timer(1./30., self.timer_callback)
    
    #def get_ground_truth_coord(self):
        
    def euclidean_distance(self, point1, waypoint):
        distance = np.sqrt((waypoint[0] - point1[0])**2 + (waypoint[1] - point1[1])**2 + (waypoint[2] - point1[2])**2)
        return distance
    
    def listener_callback_drone(self,msg):
        self.quat = msg.transform.rotation
        self.coordinate = msg.transform.translation
        self.world_coordinate = np.array([self.coordinate.x, self.coordinate.y, self.coordinate.z])
    def is_waypoint_reached(self, waypoint):
        # drone_world_coord = np.array([self.coordinate.x, self.coordinate.y, self.coordinate.z])
        return self.euclidean_distance(self.world_coordinate, waypoint) < 0.02
    
    def generate_trajectory(self):
        waypoints = []
        for i in range(self.num_points):
            theta = 2 * np.pi * i / self.num_points
            x = self.radius * np.cos(theta)
            y = self.radius * np.sin(theta)
            waypoints.append((x, y, self.height))

        self.waypoints = waypoints
    def get_next_circle_waypoint(self):
        return self.waypoints[self.index]
    def create_TrajectorySetpoint_msg(self, world_coordinates):
        msg = TrajectorySetpoint()
        #msg.raw_mode = False
        msg.position[0] = world_coordinates[0]
        msg.position[1] = world_coordinates[1]
        msg.position[2] = world_coordinates[2]
        #msg.yaw = (3.1415926 / 180.) * (float)(setpoint_yaw->value())
        msg.yaw = 0.0
        for i in range(3):
                msg.velocity[i] = 0.0
                msg.acceleration[i] = 0.0
                msg.jerk[i] = 0.0
        #msg.velocity = [0.2, 0.2, 0.2]
        msg.yawspeed = 0.0
        return msg
    def timer_callback(self):
        if(self.is_waypoint_reached(self.waypoints[self.index])) :
            self.index = self.index + 1
        else :
            msg = self.create_TrajectorySetpoint_msg(self.waypoints[self.index])
            self.publisher_.publish(msg)
        

    

