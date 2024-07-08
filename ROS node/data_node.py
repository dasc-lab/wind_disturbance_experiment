import rclpy
from rclpy.node import Node
from std_msgs.msg import *
import numpy as np
from px4_msgs.msg import TrajectorySetpoint
from foresee_msgs.msg import DynamicsData as CombinedData
from geometry_msgs.msg import TransformStamped
from rclpy.clock import Clock
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import time



class data(Node):
    def __init__(self):
        super().__init__('data')
         ###### set up parameters ######
        self.radius = 0.2
        self.height = -0.4
        self.center_x = 0.0
        self.center_y = 0.0
        self.angular_vel = 1.0
        self.g = 9.81
        self.ned_pos = None
        self.ned_vel = None
        self.ned_acc = None
        self.pos_ref = None
        self.vel_ref = None
        self.acc_ref = None
        self.coordinate_diff_3d = None
        self.coordinate_diff_2d = None
        self.euclidean_error_3d = []
        self.euclidean_error_2d = []
        self.gp_input = []
        self.gp_output = []
        self.kv = 7.4
        self.kx = 14
        self.m = 0.681
        self.thrust = None
        self.start = time.time()
        
        self.data = self.create_subscription(
		    CombinedData,
		    '/drone/combined_data',
		    self.data_callback,
		    10)
    def data_callback(self, msg):
        self.ned_pos = msg.pos
        self.ned_vel = msg.vel
        self.ned_acc = msg.acc
        self.pos_ref = msg.pos_ref
        self.vel_ref = msg.vel_ref
        self.acc_ref = msg.acc_ref
        self.calculate_thrust()
        acc_diff = self.ned_acc - self.acc_com
        self.gp_output.append(acc_diff)
        self.gp_input.append([self.ned_pos[0], self.ned_pos[1], self.ned_pos[2], self.ned_vel[0], self.ned_vel[1], self.ned_vel[2]])
        # if(time.time() - self.start) > 8:
        #     np.save('acc_diffs',self.acc_diffs)
        #     rclpy.shutdown()
    def calculate_thrust(self):
        diff_pos = np.array(self.ned_pos - self.pos_ref)
        diff_vel = np.array(self.ned_vel - self.vel_ref)
        diff_acc = np.array(self.ned_acc - self.acc_ref)
        e3 = np.array([0,0,1.]).reshape(diff_pos.shape)
        # 3D vector thrust
        self.thrust = -self.kx * diff_pos - self.kv * diff_vel + self.m * self.acc_ref #- self.m * self.g * e3 
        self.acc_com = self.thrust/self.m

    def calculate_euclidean_error(self):
        def euclidean_distance(point1, point2):
            # Ensure the points have the same dimension
            if len(point1) != len(point2):
                raise ValueError("Points must have the same dimension")
            
            # Calculate the squared differences and sum them
            squared_differences = [(p2 - p1) ** 2 for p1, p2 in zip(point1, point2)]
            sum_squared_differences = sum(squared_differences)
            
            # Calculate the distance
            distance = np.sqrt(sum_squared_differences)
            
            return distance
        
        self.coordinate_diff_3d = euclidean_distance(self.ned_pos, self.pos_ref)
        self.coordinate_diff_2d = euclidean_distance(self.ned_pos[:2],self.pos_ref[:2])
        self.euclidean_error_3d.append(self.coordinate_diff_3d)
        self.euclidean_error_2d.append(self.coordinate_diff_2d)
def main(args=None):
    rclpy.init(args=args)
    node = data()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
