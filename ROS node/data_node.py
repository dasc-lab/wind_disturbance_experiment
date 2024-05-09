import rclpy
from rclpy.node import Node
from std_msgs.msg import *
import numpy as np
from px4_msgs.msg import TrajectorySetpoint, CombinedData
from geometry_msgs.msg import TransformStamped
from rclpy.clock import Clock
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore




class data(Node):
    def __init__(self):
        super().__init__('data')
        self.
    

def main(args=None):
    rclpy.init(args=args)
    node = data()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
