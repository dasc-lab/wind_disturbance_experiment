import rclpy
from rclpy.node import Node
from std_msgs.msg import *
import numpy as np
from px4_msgs.msg import TrajectorySetpoint
from geometry_msgs.msg import TransformStamped
from rclpy.clock import Clock
import keyboard
class driveCircle(Node):
    def __init__(self):
        super().__init__('driveCircle')

        ###### set up circle parameters ######
        self.radius = 0.2
        self.height = -0.4
        self.center_x = 0.0
        self.center_y = 0.0
        self.angular_vel = 1.0

        ###### set up error parameters ######
        self.waypoint_t = None
        self.error = []
    
        ###### set up node parameters ######
        self.publisher_ = self.create_publisher(TrajectorySetpoint, '/px4_1/fmu/in/trajectory_setpoint', 10)
        self.quat = None
        self.coordinate = None
        self.world_coordinate = None
        self.clock  = self.get_clock()
        self.start_time = self.get_clock().now().nanoseconds
        #self.dt = 0.05

        ################## set up Subscription ##################
        self.timer = self.create_timer(1./80., self.timer_callback)
        self.drone_subscription = self.create_subscription(
		    TransformStamped,
		    '/px4_1/fmu/out/',
		    self.listener_callback_drone,
		    10)
    #def get_ground_truth_coord(self):
    def listener_callback_drone(self,msg):
        '''
        updates parameters
        '''
        self.quat = msg.transform.rotation
        self.coordinate = msg.transform.translation
        self.world_coordinate = np.array([self.coordinate.x, self.coordinate.y, self.coordinate.z])


    def calculate_waypoint(self):
        '''
        Calculate the waypoint in drone frame (NED)
        '''
        deltaT = (self.get_clock().now().nanoseconds-self.start_time)/10**9
        x = self.radius * np.cos(self.angular_vel * deltaT) + self.center_x
        y = self.radius * np.sin(self.angular_vel * deltaT) + self.center_y
        waypoint = [y,  x, self.height]
        self.waypoint_t = waypoint
        return waypoint

    def calculate_error(self):
        def euclidean_distance(point1, point2):
            """Calculate the Euclidean distance between two 3D points."""
            distance = np.sqrt((point1[0] - point2[0]) ** 2 +
                                (point1[1] - point2[1]) ** 2 +
                                (point1[2] - point2[2]) ** 2)
            return distance
        def calculate_ideal():
            x = self.world_coordinate[0] - self.center_x
            y = self.world_coordinate[1] - self.center_y
            theta = np.arctan2(y,x)
            ideal_x = self.radius * np.cos(theta) + self.center_x
            ideal_y = self.radius * np.sin(theta) + self.center_y
            ideal = [ideal_y, ideal_x, self.height]
            return ideal
        assert self.world_coordinate is not None
        ideal = calculate_ideal()
        return euclidean_distance(ideal, self.waypoint_t)
    
    def create_TrajectorySetpoint_msg(self):
        msg = TrajectorySetpoint()
        waypoint = self.calculate_waypoint()
        msg.position[0] = waypoint[0] #world_coordinates[0]
        msg.position[1] = waypoint[1] #world_coordinates[1]
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
        deltaT = (self.get_clock().now().nanoseconds-self.start_time)/10**9
        if deltaT > 3:
            self.error.append(self.calculate_error())

        
        
def main(args=None):
    rclpy.init(args=args)
    node = driveCircle()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
