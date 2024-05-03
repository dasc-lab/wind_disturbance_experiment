import rclpy
from rclpy.node import Node
from std_msgs.msg import *
import numpy as np
from px4_msgs.msg import TrajectorySetpoint
from rclpy.clock import Clock
class driveCircle(Node):
    def __init__(self):
        super().__init__('driveCircle')

        ###### set up circle parameters ######
        self.radius = 0.2
        self.height = -0.4
        self.center_x = 0.0
        self.center_y = 0.0
        self.angular_vel = 1.0


        ###### set up node parameters ######
        self.publisher_ = self.create_publisher(TrajectorySetpoint, '/px4_1/fmu/in/trajectory_setpoint', 10)
        self.coordinate = None
        self.quat = None
        self.world_coordinate = None
        self.clock  = self.get_clock()
        self.start_time = self.get_clock().now().nanoseconds
        #self.dt = 0.05

        ################## set up Subscription ##################
        self.timer = self.create_timer(1./30., self.timer_callback)
    
    #def get_ground_truth_coord(self):
    
    def calculate_waypoint(self):
        
       
        deltaT = (self.get_clock().now().nanoseconds-self.start_time)/10**9
        x = self.radius * np.cos(self.angular_vel * deltaT) + self.center_x
        y = self.radius * np.sin(self.angular_vel * deltaT) + self.center_y
        waypoint = [x,  y, self.height]
        return waypoint
        
    
    def create_TrajectorySetpoint_msg(self):
        msg = TrajectorySetpoint()
        waypoint = self.calculate_waypoint()
        msg.position[0] = waypoint[0] #world_coordinates[0]
        msg.position[1] = waypoint[1]#world_coordinates[1]
        msg.position[2] = self.height #world_coordinates[2]
        #msg.yaw = (3.1415926 / 180.) * (float)(setpoint_yaw->value())
        msg.yaw = 290 * 3.14/180.0 #0.0
        for i in range(3):
            msg.velocity[i] = 0.0
            msg.acceleration[i] = 0.0
            msg.jerk[i] = 0.0
        #msg.velocity = [0.2, 0.2, 0.2]
        msg.yawspeed = 0.0
        return msg
    
    def timer_callback(self):
        msg = self.create_TrajectorySetpoint_msg()
        self.publisher_.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)

    node = driveCircle()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
