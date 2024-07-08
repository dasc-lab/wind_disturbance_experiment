import rclpy
from rclpy.node import Node
from std_msgs.msg import *
import numpy as np
from px4_msgs.msg import TrajectorySetpoint, VehicleLocalPosition
from foresee_msgs.msg import DynamicsData as CombinedData
from geometry_msgs.msg import TransformStamped
from rclpy.clock import Clock
import time

class message_node(Node):
    def __init__(self):
        super().__init__('message_node')
        
        ###### set up parameters ######
        self.radius = 0.2
        self.height = -0.4
        self.center_x = 0.0
        self.center_y = 0.0
        self.angular_vel = 1.0
        self.ned_pos = None
        self.ned_vel = None
        self.ned_acc = None
        self.pos_ref = None
        self.vel_ref = None
        self.acc_ref = None
        self.acc_com = None
        self.coordinate_diff_3d = None
        self.coordinate_diff_2d = None
        self.euclidean_error_3d = []
        self.euclidean_error_2d = []
        
        ############ set up publisher ############
        self.publisher_ = self.create_publisher(CombinedData, '/drone/combined_data', 10)
        
    
        ################## set up Subscription ##################

        #self.timer = self.create_timer(1./80., self.timer_callback)
        self.actual = self.create_subscription(
		    VehicleLocalPosition,
		    '/px4_1/fmu/out/vehicle_local_position',
		    self.coordinate_callback,
		    10)
        
        self.ref = self.create_subscription(
		    TrajectorySetpoint,
		    '/px4_1/fmu/in/TrajectorySetpoint',
		    self.reference_callback,
		    10)
        
    def create_CombinedData_msg(self):
        msg = CombinedData()
        msg.pos = self.ned_pos
        msg.vel = self.ned_vel
        msg.acc = self.ned_acc
        msg.pos_ref = self.pos_ref
        msg.vel_ref = self.vel_ref
        msg.acc_ref = self.acc_ref
        return msg
        ################## set up call backs ##################
        
    def coordinate_callback(self, msg):
        self.ned_pos = [msg.x,msg.y,msg.z]
        self.ned_vel = [msg.vx,msg.vy,msg.vz]
        self.ned_acc = [msg.ax, msg.ay, msg.az]
        message = self.create_CombinedData_msg()
        self.publisher_.publish(message)
            
    def reference_callback(self, msg):
        self.pos_ref = msg.position
        self.vel_ref = msg.velocity
        self.acc_ref = msg.acceleration

    #def get_ground_truth_coord(self):
    
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

    node = message_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
