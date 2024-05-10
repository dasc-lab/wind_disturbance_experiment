import rclpy
from rclpy.node import Node
from std_msgs.msg import *
import numpy as np
from px4_msgs.msg import TrajectorySetpoint, CombinedData
from geometry_msgs.msg import TransformStamped
from rclpy.clock import Clock

class message(Node):
    def __init__(self):
        super().__init__('message')
        
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
        ############ set up publisher ############
        self.publisher_ = self.create_publisher(CombinedData, '/drone/combined_data', 10)
        
    
        ################## set up Subscription ##################

        #self.timer = self.create_timer(1./80., self.timer_callback)
        self.actual = self.create_subscription(
		    TransformStamped,
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
            message = create_CombinedData_msg()
            self.publisher.publish(message)
             
        def reference_callback(self, msg):
            self.pos_ref = msg.position
            self.vel_ref = msg.velocity
            self.acc_ref = msg.acceleration

    #def get_ground_truth_coord(self):
    
    
    
def main(args=None):
    rclpy.init(args=args)

    node = message()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
